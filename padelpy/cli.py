"""Command line interface for PaDELPy - molecular descriptor calculations."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Union

import click
import pandas as pd
import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table

from .functions import from_smiles, from_sdf, from_mdl
from .wrapper import padeldescriptor

console = Console()


def print_version(ctx, param, value):
    """Print version information."""
    if not value or ctx.resilient_parsing:
        return
    
    try:
        from importlib.metadata import version
        padelpy_version = version('padelpy')
    except ImportError:
        padelpy_version = "Unknown"
    
    click.echo(f"PaDELPy v{padelpy_version}")
    click.echo(f"Python: {sys.version.split()[0]}")
    click.echo(f"Platform: {sys.platform}")
    ctx.exit()


@click.group()
@click.option('--version', is_flag=True, callback=print_version,
              expose_value=False, is_eager=True, help='Show version and exit.')
@click.pass_context
def cli(ctx):
    """PaDELPy - Python wrapper for PaDEL-Descriptor molecular descriptors.
    
    Calculate molecular descriptors and fingerprints using the PaDEL-Descriptor
    Java software package. Supports SMILES strings, SDF files, and MDL files.
    """
    ctx.ensure_object(dict)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--output", 
    type=click.Path(path_type=Path),
    help="Output CSV file for descriptors. If not specified, prints to stdout."
)
@click.option(
    "--format", "input_format",
    type=click.Choice(['smi', 'smiles', 'sdf', 'mol'], case_sensitive=False),
    help="Input file format (auto-detected if not specified)"
)
@click.option(
    "--output-format",
    type=click.Choice(['csv', 'json', 'table'], case_sensitive=False),
    default='csv',
    help="Output format (default: csv)"
)
@click.option(
    "--descriptors/--no-descriptors",
    default=True,
    help="Calculate molecular descriptors (default: True)"
)
@click.option(
    "--fingerprints/--no-fingerprints",
    default=False,
    help="Calculate molecular fingerprints (default: False)"
)
@click.option(
    "--timeout",
    type=int,
    default=60,
    help="Timeout per molecule in seconds (default: 60)"
)
@click.option(
    "--maxruntime",
    type=int,
    default=-1,
    help="Maximum runtime per molecule in seconds (default: -1, unlimited)"
)
@click.option(
    "--threads",
    type=int,
    default=-1,
    help="Number of threads to use (default: -1, all available)"
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Verbose output"
)
def calculate(
    input_file: Path,
    output: Optional[Path],
    input_format: Optional[str],
    output_format: str,
    descriptors: bool,
    fingerprints: bool,
    timeout: int,
    maxruntime: int,
    threads: int,
    verbose: bool
):
    """Calculate molecular descriptors for molecules in input file.
    
    Examples:
    \b
        padelpy calculate molecules.smi --output descriptors.csv
        padelpy calculate compounds.sdf --fingerprints --output results.csv
        padelpy calculate data.smi --output-format json --descriptors --fingerprints
    """
    if verbose:
        console.print(f"[green]Reading molecules from:[/green] {input_file}")
    
    # Auto-detect format
    if not input_format:
        suffix = input_file.suffix.lower()
        if suffix in ['.smi', '.smiles']:
            input_format = 'smiles'
        elif suffix == '.sdf':
            input_format = 'sdf'
        elif suffix == '.mol':
            input_format = 'mol'
        else:
            input_format = 'smiles'  # Default assumption
    
    if not descriptors and not fingerprints:
        console.print("[red]Error:[/red] Must calculate either descriptors or fingerprints")
        sys.exit(1)
    
    try:
        # Read input file
        if input_format.lower() in ['smi', 'smiles']:
            smiles_list = read_smiles_file(input_file, verbose)
            if not smiles_list:
                console.print("[red]Error:[/red] No valid SMILES found in input file")
                sys.exit(1)
            
            if verbose:
                console.print(f"[green]Loaded {len(smiles_list)} SMILES[/green]")
                console.print("[yellow]Calculating descriptors...[/yellow]")
            
            # Calculate descriptors for all SMILES
            results = from_smiles(
                smiles_list,
                descriptors=descriptors,
                fingerprints=fingerprints,
                timeout=timeout,
                maxruntime=maxruntime,
                threads=threads
            )
            
        elif input_format.lower() == 'sdf':
            if verbose:
                console.print("[yellow]Calculating descriptors from SDF file...[/yellow]")
            
            # For SDF files, use the direct wrapper
            if output:
                output_csv = str(output) if output.suffix.lower() == '.csv' else f"{output}.csv"
            else:
                import tempfile
                output_csv = tempfile.mktemp(suffix='.csv')
            
            results = from_sdf(
                str(input_file),
                output_csv=output_csv,
                descriptors=descriptors,
                fingerprints=fingerprints,
                timeout=timeout,
                maxruntime=maxruntime,
                threads=threads
            )
            
        else:
            console.print(f"[red]Error:[/red] Unsupported format '{input_format}'")
            sys.exit(1)
        
        if verbose:
            if isinstance(results, list):
                console.print(
                    f"[green]Successfully calculated descriptors for "
                    f"{len(results)} molecules[/green]"
                )
            else:
                console.print(f"[green]Successfully calculated descriptors[/green]")
        
        # Output results
        if output:
            save_results(results, output, output_format, verbose)
        else:
            display_results(results, output_format)
    
    except Exception as e:
        console.print(f"[red]Error calculating descriptors:[/red] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.argument("smiles_list", nargs=-1, required=True)
@click.option(
    "--output-format",
    type=click.Choice(['csv', 'json', 'table'], case_sensitive=False),
    default='table',
    help="Output format (default: table)"
)
@click.option(
    "--descriptors/--no-descriptors",
    default=True,
    help="Calculate molecular descriptors (default: True)"
)
@click.option(
    "--fingerprints/--no-fingerprints",
    default=False,
    help="Calculate molecular fingerprints (default: False)"
)
@click.option(
    "--timeout",
    type=int,
    default=60,
    help="Timeout per molecule in seconds (default: 60)"
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Verbose output"
)
def single(
    smiles_list: tuple[str, ...],
    output_format: str,
    descriptors: bool,
    fingerprints: bool,
    timeout: int,
    verbose: bool
):
    """Calculate descriptors for SMILES strings provided as arguments.
    
    Examples:
    \b
        padelpy single "CCO" "CCC" "c1ccccc1"
        padelpy single "CCO" --fingerprints --output-format json
        padelpy single "c1ccccc1" --descriptors --fingerprints
    """
    if not descriptors and not fingerprints:
        console.print("[red]Error:[/red] Must calculate either descriptors or fingerprints")
        sys.exit(1)
    
    try:
        if verbose:
            console.print(
                f"[yellow]Calculating descriptors for {len(smiles_list)} "
                f"molecules...[/yellow]"
            )
        
        results = from_smiles(
            list(smiles_list),
            descriptors=descriptors,
            fingerprints=fingerprints,
            timeout=timeout
        )
        
        if verbose:
            console.print(f"[green]Successfully calculated descriptors[/green]")
        
        display_results(results, output_format)
        
    except Exception as e:
        console.print(f"[red]Error calculating descriptors:[/red] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format", "input_format",
    type=click.Choice(['smi', 'smiles', 'sdf', 'mol'], case_sensitive=False),
    help="Input file format (auto-detected if not specified)"
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Verbose output"
)
def validate(
    input_file: Path,
    input_format: Optional[str],
    verbose: bool
):
    """Validate molecules in input file for descriptor calculations.
    
    Examples:
    \b
        padelpy validate molecules.smi
        padelpy validate structures.sdf --verbose
    """
    if verbose:
        console.print(f"[green]Validating molecules in:[/green] {input_file}")
    
    # Auto-detect format
    if not input_format:
        suffix = input_file.suffix.lower()
        if suffix in ['.smi', '.smiles']:
            input_format = 'smiles'
        elif suffix == '.sdf':
            input_format = 'sdf'
        elif suffix == '.mol':
            input_format = 'mol'
        else:
            input_format = 'smiles'
    
    try:
        if input_format.lower() in ['smi', 'smiles']:
            smiles_list = read_smiles_file(input_file, verbose)
            
            if verbose:
                console.print(f"[green]Found {len(smiles_list)} SMILES strings[/green]")
            
            valid_count = len([s for s in smiles_list if s.strip()])
            
            console.print(f"\n[yellow]Validation Results:[/yellow]")
            console.print(f"Total SMILES: {len(smiles_list)}")
            console.print(f"Non-empty: {valid_count}")
            
            if len(smiles_list) != valid_count:
                missing = len(smiles_list) - valid_count
                console.print(
                    f"[yellow]Warning:[/yellow] {missing} empty entries found"
                )
        
        elif input_format.lower() == 'sdf':
            # For SDF validation, try reading with PaDEL
            console.print("[yellow]SDF validation requires Java runtime...[/yellow]")
            console.print(f"File: {input_file}")
            console.print("Use 'padelpy calculate' to test SDF processing.")
        
        else:
            console.print(f"[red]Error:[/red] Unsupported format '{input_format}'")
            sys.exit(1)
    
    except Exception as e:
        console.print(f"[red]Error validating file:[/red] {e}")
        sys.exit(1)


@cli.command()
def descriptors():
    """Show information about available descriptors and fingerprints."""
    
    console.print("[bold]PaDEL-Descriptor Available Calculations[/bold]")
    console.print("=" * 50)
    console.print()
    
    console.print("[yellow]Molecular Descriptors:[/yellow]")
    console.print("  • 1D Descriptors: Atom and bond counts")
    console.print("  • 2D Descriptors: Topological and graph-based")
    console.print("  • 3D Descriptors: Geometric and surface descriptors")
    console.print("  • Total: ~1800+ descriptors available")
    console.print()
    
    console.print("[yellow]Molecular Fingerprints:[/yellow]")
    console.print("  • AtomPairs2D: Atom pair fingerprints")
    console.print("  • AtomPairs2DCount: Atom pair count fingerprints")
    console.print("  • EState: Electrotopological state fingerprints")
    console.print("  • CDK: Chemistry Development Kit fingerprints")
    console.print("  • CDKextended: Extended CDK fingerprints")
    console.print("  • CDKgraphonly: Graph-only CDK fingerprints")
    console.print("  • KlekotaRoth: Klekota-Roth fingerprints")
    console.print("  • KlekotaRothCount: Klekota-Roth count fingerprints")
    console.print("  • MACCS: MACCS 166-bit fingerprints")
    console.print("  • PubChem: PubChem fingerprints")
    console.print("  • SubstructureCount: Substructure count fingerprints")
    console.print()
    
    console.print("[yellow]Usage Examples:[/yellow]")
    examples = [
        ("Descriptors only", "padelpy single 'CCO' --descriptors --no-fingerprints"),
        ("Fingerprints only", "padelpy single 'CCO' --no-descriptors --fingerprints"),
        ("Both", "padelpy single 'CCO' --descriptors --fingerprints"),
        ("From file", "padelpy calculate molecules.smi --output results.csv"),
    ]
    
    table = Table()
    table.add_column("Calculation", style="cyan")
    table.add_column("Command", style="white")
    
    for desc, cmd in examples:
        table.add_row(desc, cmd)
    
    console.print(table)


@cli.command()
def info():
    """Display system and package information."""
    try:
        from importlib.metadata import version
        padelpy_version = version('padelpy')
    except ImportError:
        padelpy_version = "Unknown"
    
    console.print("[bold]PaDELPy Information[/bold]")
    console.print("=" * 30)
    console.print(f"Version: {padelpy_version}")
    console.print(f"Python: {sys.version.split()[0]}")
    console.print(f"Platform: {sys.platform}")
    console.print("")
    
    # Check dependencies
    deps = {
        'NumPy': 'numpy',
        'Pandas': 'pandas',
        'Click': 'click',
        'Rich': 'rich'
    }
    
    console.print("[yellow]Dependencies:[/yellow]")
    for name, module in deps.items():
        try:
            from importlib.metadata import version
            version_str = version(module)
            console.print(f"  {name}: {version_str}")
        except ImportError:
            console.print(f"  {name}: [red]Not installed[/red]")
    
    # Check Java
    console.print("")
    console.print("[yellow]Java Runtime:[/yellow]")
    try:
        import subprocess
        result = subprocess.run(['java', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            java_version = result.stderr.split('\n')[0] if result.stderr else "Unknown"
            console.print(f"  Java: {java_version}")
        else:
            console.print("  Java: [red]Not found or not working[/red]")
    except Exception:
        console.print("  Java: [red]Not found[/red]")
    
    console.print("")
    console.print("[yellow]Features:[/yellow]")
    console.print("  • 1800+ molecular descriptors")
    console.print("  • Multiple fingerprint types")
    console.print("  • SMILES, SDF, and MDL file support")
    console.print("  • Multi-threaded calculations")
    console.print("")
    console.print("[yellow]References:[/yellow]")
    console.print("  PaDEL-Descriptor: Yap CW (2011) J Comput Chem 32:1466-1474")
    console.print("  DOI: 10.1002/jcc.21707")


def read_smiles_file(input_file: Path, verbose: bool = False) -> List[str]:
    """Read SMILES from input file."""
    smiles_list = []
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    if verbose:
        lines = track(lines, description="Reading SMILES...")
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if line and not line.startswith('#'):
            try:
                # Handle both SMILES and SMILES + name format
                parts = line.split('\t') if '\t' in line else line.split()
                smiles = parts[0]
                smiles_list.append(smiles)
            except Exception as e:
                if verbose:
                    console.print(f"[yellow]Warning:[/yellow] Error reading line {line_num}: {e}")
    
    return smiles_list


def save_results(results: Union[dict, List[dict]], output_path: Path, 
                output_format: str, verbose: bool = False):
    """Save results to file."""
    try:
        if output_format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        else:  # CSV format
            if isinstance(results, list):
                df = pd.DataFrame(results)
            else:
                df = pd.DataFrame([results])
            df.to_csv(output_path, index=False)
        
        if verbose:
            console.print(f"[green]Results saved to:[/green] {output_path}")
    
    except Exception as e:
        console.print(f"[red]Error saving results:[/red] {e}")


def display_results(results: Union[dict, List[dict]], output_format: str):
    """Display results to stdout."""
    if output_format == 'json':
        click.echo(json.dumps(results, indent=2, default=str))
    
    elif output_format == 'csv':
        if isinstance(results, list):
            df = pd.DataFrame(results)
        else:
            df = pd.DataFrame([results])
        click.echo(df.to_csv(index=False))
    
    else:  # table format
        if isinstance(results, dict):
            results = [results]
        
        if not results:
            console.print("[yellow]No results to display[/yellow]")
            return
        
        # Show first few descriptors for brevity
        table = Table(title="Molecular Descriptors (Sample)")
        table.add_column("SMILES/ID", style="cyan", max_width=20)
        
        # Get first few descriptor names
        first_result = results[0]
        descriptor_names = list(first_result.keys())[:10]  # Show first 10
        
        for desc_name in descriptor_names:
            table.add_column(desc_name, style="white", justify="right", max_width=10)
        
        for i, result in enumerate(results[:20]):  # Show first 20 molecules
            row = [f"Mol_{i+1}"]
            for desc_name in descriptor_names:
                value = result.get(desc_name, "N/A")
                if isinstance(value, float):
                    row.append(f"{value:.4f}")
                else:
                    row.append(str(value)[:8])
            table.add_row(*row)
        
        console.print(table)
        
        if len(results) > 20:
            console.print(f"[yellow]... and {len(results) - 20} more molecules[/yellow]")
        
        if len(descriptor_names) < len(first_result):
            total_descriptors = len(first_result)
            console.print(f"[yellow]Showing 10 of {total_descriptors} descriptors. Use --output-format csv/json to see all.[/yellow]")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()