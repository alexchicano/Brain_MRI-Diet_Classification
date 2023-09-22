import tempfile
import shutil
import subprocess
import os
import nibabel as nib
import numpy as np
import concurrent
from concurrent.futures.thread import ThreadPoolExecutor

def parallel_run(
    func, 
    args, 
    num_threads
):
    """Runs func in parallel with the given args and returns an ordered list with the returned values.

    args MUST be a list of lists. args is a list with one element per run instance, and each element is 
    a list of the arguments for the function in that run instance.

    When all run instances are finished, it returns a list with the returned elements of the function
    """

    # Assert list of lists to comply with variadic positional arguments (i.e. the * in fn(*args))
    assert all([isinstance(arg, list) or isinstance(arg, tuple) or isinstance(arg, dict) for arg in args]), \
        'Function arguments must be given as tuple, list or dictionary'
    assert callable(func), 'func must be a callable function'
    # Define output variable and load function wrapper to maintain correct list order
    results = [None] * len(args)
    def _run_func(n_, args_):
        if isinstance(args_, list) or isinstance(args_, tuple):
            results[n_] = func(*args_)
        elif isinstance(args_, dict):
            results[n_] = func(**args_)
        else:
            raise ValueError('Function arguments were not tuple, list or dict')
    # Parallel run func and store the results in the right place
    pool = ThreadPoolExecutor(max_workers=num_threads)
    future_tasks = [pool.submit(_run_func, n, args) for n, args in enumerate(args)]
    # Check if any exceptions occured during execution
    for n, future_task in enumerate(future_tasks):
        concurrent.futures.wait([future_task])
        future_task.result()
    pool.shutdown(wait=True)
    return results

def remove_ext(filepath):
    """Removes all extensions of the filename pointed by filepath.

    :Example:

    >>> remove_ext('home/user/t1_image.nii.gz')
    'home/user/t1_image'
    """
    paths = filepath.split('/')
    return filepath if '.' not in paths[-1] else os.sep.join(paths[:-1] + [paths[-1].split('.')[0]])


def run_fsl_bet(
    t1_fp, 
    brain_out_fp=None, 
    brain_mask_out_fp=None, 
    skull_out_fp=None, 
    betopts='-B', 
    fsldir='/home/alex/fsl'
):
    """
    useful betopts (mutually exclusive options, so just one of these at a time)
      -R          robust brain centre estimation (iterates BET several times)
      -S          eye & optic nerve cleanup (can be useful in SIENA - disables -o option)
      -B          bias field & neck cleanup (can be useful in SIENA)
    
    check outputs to see which option is best
    """

    with tempfile.TemporaryDirectory() as tempdir:
        t1_fp_in = os.path.join(tempdir, 't1.nii.gz')
        brain_fp = os.path.join(tempdir, 't1_bet_brain.nii.gz')
        brain_mask_fp = os.path.join(tempdir, 't1_bet_brain_mask.nii.gz')
        skull_fp = os.path.join(tempdir, 't1_bet_brain_skull.nii.gz')

        shutil.copy(t1_fp, t1_fp_in)
        subprocess.check_output(['bash', '-c', f'{fsldir}/bin/bet {t1_fp_in} t1_bet_brain -s {betopts}'], cwd=tempdir)

        if brain_out_fp is not None:
            shutil.copy(brain_fp, brain_out_fp)

        if brain_mask_out_fp is not None:
            shutil.copy(brain_mask_fp, brain_mask_out_fp)

        if skull_out_fp is not None:
            shutil.copy(skull_fp, skull_out_fp)


def run_fsl_pairreg(
    ref_filepath, 
    mov_filepath, 
    transform_filepath, 
    workdir=None, 
    fsldir='/home/alex/fsl'
):
    """
    workdir: can be used to specify a folder where the intermediate results are stored, otherwise they are
        stored in a temporary directory which is erased at the end
    """
    tempdir = None
    if workdir is None:
        tempdir = tempfile.TemporaryDirectory()
        workdir = tempdir.name

    try:
        ref_brain_fp = os.path.join(workdir, 'ref_bet_brain.nii.gz')
        ref_skull_fp = os.path.join(workdir, 'ref_bet_brain_skull.nii.gz')
        run_fsl_bet(ref_filepath, brain_out_fp=ref_brain_fp, skull_out_fp=ref_skull_fp, betopts='-R -S')

        mov_brain_fp = os.path.join(workdir, 'mov_bet_brain.nii.gz')
        mov_skull_fp = os.path.join(workdir, 'mov_bet_brain_skull.nii.gz')
        run_fsl_bet(mov_filepath, brain_out_fp=mov_brain_fp, skull_out_fp=mov_skull_fp, betopts='-R -S')

        subprocess.check_output(
            ['bash', '-c', f'{fsldir}/bin/pairreg {ref_brain_fp} {mov_brain_fp} {ref_skull_fp} {mov_skull_fp} {transform_filepath}'])
    finally:
        if tempdir is not None:
            tempdir.cleanup()

def apply_fsl_transform(
    filepath_in, 
    filepath_ref, 
    filepath_out, 
    filepath_transform, 
    fsldir='/home/alex/fsl'
):
    tx_cmd = '{}/bin/flirt -out {} -applyxfm -init {} -ref {} -in {}'.format(
        fsldir, filepath_out, filepath_transform, filepath_ref, filepath_in)
    subprocess.check_output(['bash', '-c', tx_cmd])


def invert_fsl_transform(
    filepath_transform, 
    filepath_inverse, 
    fsldir='/home/alex/fsl'
):
    subprocess.check_output(
        ['bash', '-c', f'{fsldir}/bin/convert_xfm -inverse -omat {filepath_inverse} {filepath_transform}'])

def segment_anat_fast_first(
        fpath_in,
        is_bias_corrected,
        fpaths_fast,
        fpath_first,
        fpath_anat_to_mni_out,
        fp_skull_stripped_out,
        fp_nonl_skull_stripped_out,
        decimals=4, # Rounding to 4 decimals saves hard drive space 
        remove_anat_dir=True
):
    """Runs fsl_anat pipeline doing skull-stripping, tissue segmentation (FAST) and subcortical structures (FIRST)

    fpaths_fast is a 4 element list, with the filepaths for the background, CSF, GM and WM output probability maps
    fpath_first is a single string
    """
    
    assert os.path.isfile(fpath_in), fpath_in

    anat_dir = remove_ext(fpath_in) + '.anat'
    fpaths_anat_fast = [os.path.join(anat_dir, 'T1_fast_pve_{}.nii.gz'.format(i)) for i in range(3)]
    fpath_anat_first = os.path.join(anat_dir, 'T1_subcort_seg.nii.gz')

    # Build and run fsl_anat command 
    bias_opt = '--nobias' if is_bias_corrected else '--weakbias'
    subprocess.check_output(
        ['bash', '-c', f'fsl_anat --clobber --nocrop --noreorient  {bias_opt} -i {fpath_in}'])

    # Save "T1_to_MNI_nonlin.nii.gz" and "MNI152_T1_2mm_brain_mask_dil1.nii.gz" to output folder:
    fpath_anat_to_mni = os.path.join(anat_dir, 'T1_to_MNI_nonlin.nii.gz')
    fpath_mni_brain_mask = os.path.join(anat_dir, 'MNI152_T1_2mm_brain_mask_dil1.nii.gz')
    shutil.copy(
        fpath_anat_to_mni, 
        fpath_anat_to_mni_out
    )
    shutil.copy(
        fpath_mni_brain_mask, 
        fp_nonl_skull_stripped_out
    )
    
    
    # Load FAST segmentations, add background prob channel
    fast_nifti = nib.load(fpaths_anat_fast[0])
    fast_pves = [nib.load(fp).get_fdata() for fp in fpaths_anat_fast]
    fast_pves.insert(0, 1.0 - np.sum(np.stack(fast_pves, axis=0), axis=0))  # Background probability

    # Save fast segmentation by itself
    for fast_pve, fpath_tgt_fast in zip(fast_pves, fpaths_fast):
        fast_pve_arr = np.round(fast_pve, decimals=decimals)
        nib.Nifti1Image(fast_pve_arr, fast_nifti.affine, fast_nifti.header).to_filename(fpath_tgt_fast)

    # Load first segmentation and store
    first_nifti = nib.load(fpath_anat_first)
    first_seg = first_nifti.get_fdata().astype(int)
    nib.Nifti1Image(first_seg, fast_nifti.affine, fast_nifti.header).to_filename(fpath_first)
    
    # Get skull stripped image out
    shutil.copy(
        anat_dir + '/T1_biascorr_brain.nii.gz',
        fp_skull_stripped_out)
    
    if remove_anat_dir:
        shutil.rmtree(anat_dir)

