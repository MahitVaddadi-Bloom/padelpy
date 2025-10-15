"""Enhanced utility functions for PaDELPy with NumPy and Pandas support."""

from typing import Union, List, Dict, Any, Optional
import warnings
import pandas as pd
import numpy as np

from .functions import from_smiles, from_sdf, from_mdl

__all__ = [
    "calculate_descriptors_array",
    "calculate_descriptors_dataframe", 
    "descriptors_to_numpy",
    "batch_calculate",
]


def calculate_descriptors_array(
    smiles: Union[str, List[str]],
    descriptors: bool = True,
    fingerprints: bool = False,
    timeout: int = 60,
    fill_missing: float = 0.0
) -> np.ndarray:
    """Calculate descriptors and return as NumPy array.
    
    Args:
        smiles: SMILES string(s) to process
        descriptors: Calculate molecular descriptors
        fingerprints: Calculate molecular fingerprints  
        timeout: Timeout per molecule in seconds
        fill_missing: Value to fill missing/NaN entries
        
    Returns:
        np.ndarray: Descriptors as NumPy array (molecules x descriptors)
    """
    if isinstance(smiles, str):
        smiles = [smiles]
    
    results = from_smiles(
        smiles,
        descriptors=descriptors,
        fingerprints=fingerprints,
        timeout=timeout
    )
    
    return descriptors_to_numpy(results, fill_missing=fill_missing)


def calculate_descriptors_dataframe(
    smiles: Union[str, List[str]],
    descriptors: bool = True,
    fingerprints: bool = False,
    timeout: int = 60,
    include_smiles: bool = True
) -> pd.DataFrame:
    """Calculate descriptors and return as Pandas DataFrame.
    
    Args:
        smiles: SMILES string(s) to process
        descriptors: Calculate molecular descriptors
        fingerprints: Calculate molecular fingerprints
        timeout: Timeout per molecule in seconds
        include_smiles: Include SMILES column in output
        
    Returns:
        pd.DataFrame: Descriptors as DataFrame
    """
    if isinstance(smiles, str):
        smiles = [smiles]
    
    results = from_smiles(
        smiles,
        descriptors=descriptors,
        fingerprints=fingerprints,
        timeout=timeout
    )
    
    df = pd.DataFrame(results)
    
    if include_smiles:
        # Add SMILES column at the beginning
        df.insert(0, 'SMILES', smiles[:len(results)])
    
    return df


def descriptors_to_numpy(
    descriptors: Union[Dict[str, Any], List[Dict[str, Any]]],
    fill_missing: float = 0.0,
    dtype: Optional[np.dtype] = None
) -> np.ndarray:
    """Convert descriptor dictionaries to NumPy array.
    
    Args:
        descriptors: Descriptor dictionary or list of dictionaries
        fill_missing: Value to fill missing/NaN entries
        dtype: NumPy data type for output array
        
    Returns:
        np.ndarray: Descriptors as NumPy array
    """
    if isinstance(descriptors, dict):
        descriptors = [descriptors]
    
    if not descriptors:
        return np.array([])
    
    # Get all possible descriptor names
    all_keys = set()
    for desc_dict in descriptors:
        all_keys.update(desc_dict.keys())
    
    # Remove non-numeric keys that might be present
    non_numeric_keys = {'Name', 'SMILES', 'ID'}
    numeric_keys = sorted(all_keys - non_numeric_keys)
    
    # Create array
    if dtype is None:
        dtype = np.float64
    
    array = np.full((len(descriptors), len(numeric_keys)), fill_missing, dtype=dtype)
    
    for i, desc_dict in enumerate(descriptors):
        for j, key in enumerate(numeric_keys):
            value = desc_dict.get(key, fill_missing)
            try:
                # Handle various numeric formats
                if value == '' or value is None:
                    array[i, j] = fill_missing
                elif isinstance(value, str):
                    if value.lower() in ['nan', 'inf', '-inf']:
                        array[i, j] = fill_missing
                    else:
                        array[i, j] = float(value)
                else:
                    array[i, j] = float(value)
            except (ValueError, TypeError):
                array[i, j] = fill_missing
                warnings.warn(f"Could not convert '{value}' to float for descriptor '{key}'")
    
    return array


def batch_calculate(
    smiles_list: List[str],
    batch_size: int = 100,
    descriptors: bool = True,
    fingerprints: bool = False,
    timeout: int = 60,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """Calculate descriptors in batches for large datasets.
    
    Args:
        smiles_list: List of SMILES strings
        batch_size: Number of molecules per batch
        descriptors: Calculate molecular descriptors
        fingerprints: Calculate molecular fingerprints
        timeout: Timeout per molecule in seconds
        verbose: Print progress information
        
    Returns:
        List[Dict]: Combined results from all batches
    """
    all_results = []
    num_batches = (len(smiles_list) + batch_size - 1) // batch_size
    
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        if verbose:
            print(f"Processing batch {batch_num}/{num_batches} ({len(batch)} molecules)...")
        
        try:
            batch_results = from_smiles(
                batch,
                descriptors=descriptors,
                fingerprints=fingerprints,
                timeout=timeout
            )
            all_results.extend(batch_results)
            
            if verbose:
                print(f"  ✓ Batch {batch_num} completed successfully")
                
        except Exception as e:
            if verbose:
                print(f"  ✗ Batch {batch_num} failed: {e}")
            # Add placeholder results for failed batch
            for _ in batch:
                all_results.append({})
    
    return all_results


def validate_numpy_compatibility():
    """Validate NumPy version compatibility.
    
    Returns:
        bool: True if NumPy version is compatible
    """
    try:
        import numpy as np
        
        # Test basic array operations that should work in both NumPy 1.x and 2.x
        test_array = np.array([1.0, 2.0, 3.0])
        
        # Test operations that might have changed between versions
        _ = np.full((3, 3), 0.0)
        _ = test_array.astype(np.float64)
        _ = np.isnan(test_array)
        
        return True
        
    except Exception as e:
        warnings.warn(f"NumPy compatibility issue: {e}")
        return False


def validate_pandas_compatibility():
    """Validate Pandas version compatibility.
    
    Returns:
        bool: True if Pandas version is compatible
    """
    try:
        import pandas as pd
        
        # Test basic DataFrame operations
        test_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        _ = test_df.to_csv(index=False)
        _ = test_df.fillna(0)
        
        return True
        
    except Exception as e:
        warnings.warn(f"Pandas compatibility issue: {e}")
        return False


# Validate compatibility on import
if not validate_numpy_compatibility():
    warnings.warn("NumPy compatibility issues detected. Some features may not work correctly.")

if not validate_pandas_compatibility():
    warnings.warn("Pandas compatibility issues detected. Some features may not work correctly.")