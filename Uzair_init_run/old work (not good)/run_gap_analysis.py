"""
Master Execution Script for SHPDE Gap Analysis

This script orchestrates all four experiments and generates a consolidated report.

Author: Gap Analysis Framework
Date: January 26, 2026
"""

import os
import sys
import subprocess
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
EXPERIMENTS = [
    {
        'id': 1,
        'name': 'Threshold Sensitivity Analysis',
        'script': 'experiment_1_threshold_sweep.py',
        'output_dir': 'experiment_1_results'
    },
    {
        'id': 2,
        'name': 'Instruction Momentum Analysis',
        'script': 'experiment_2_instruction_momentum.py',
        'output_dir': 'experiment_2_results'
    },
    {
        'id': 3,
        'name': 'Semantic Drift Analysis',
        'script': 'experiment_3_semantic_drift.py',
        'output_dir': 'experiment_3_results'
    },
    {
        'id': 4,
        'name': 'Patch Inefficacy Analysis',
        'script': 'experiment_4_patch_inefficacy.py',
        'output_dir': 'experiment_4_results'
    }
]

# --- BANNER ---
def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║        SELF-HEALING PROMPT DEFENSE ENGINE (SHPDE)                       ║
║                   GAP ANALYSIS FRAMEWORK                                ║
║                                                                          ║
║  Comprehensive Boundary Condition Testing & Vulnerability Detection     ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

Experiments:
  1. Adversarial Sensitivity (τ-Threshold Dead Zone)
  2. Context Priority Conflict (Instruction Momentum)
  3. Semantic Drift Gaps (Low-Energy/High-Intent Attacks)
  4. Patch Inefficacy (Defense Cancellation)

Starting execution at: {timestamp}

"""
    print(banner.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# --- EXECUTION ---
def run_experiment(exp):
    """Run a single experiment."""
    print(f"\n{'='*80}")
    print(f"🚀 EXPERIMENT {exp['id']}: {exp['name']}")
    print(f"{'='*80}\n")
    
    try:
        # Run the experiment script
        result = subprocess.run(
            [sys.executable, exp['script']],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout per experiment
        )
        
        if result.returncode == 0:
            print(f"✅ Experiment {exp['id']} completed successfully!")
            print(f"📁 Results saved to: {exp['output_dir']}/")
            return True
        else:
            print(f"❌ Experiment {exp['id']} failed!")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  Experiment {exp['id']} timed out after 30 minutes!")
        return False
    except Exception as e:
        print(f"❌ Experiment {exp['id']} encountered an error: {str(e)}")
        return False

def run_specific_experiment(exp_id):
    """Run a specific experiment by ID."""
    for exp in EXPERIMENTS:
        if exp['id'] == exp_id:
            return run_experiment(exp)
    print(f"❌ Experiment {exp_id} not found!")
    return False

def run_all_experiments():
    """Run all experiments sequentially."""
    results = {}
    
    for exp in EXPERIMENTS:
        success = run_experiment(exp)
        results[exp['id']] = success
    
    return results

def generate_consolidated_report(results):
    """Generate a consolidated report from all experiments."""
    print(f"\n{'='*80}")
    print("📊 GENERATING CONSOLIDATED REPORT")
    print(f"{'='*80}\n")
    
    report = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║           SHPDE GAP ANALYSIS - CONSOLIDATED FINDINGS REPORT             ║
╚══════════════════════════════════════════════════════════════════════════╝

Execution Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{'='*80}
EXPERIMENT EXECUTION STATUS
{'='*80}

"""
    
    for exp in EXPERIMENTS:
        status = "✅ SUCCESS" if results.get(exp['id'], False) else "❌ FAILED"
        report += f"  Experiment {exp['id']}: {exp['name']} - {status}\n"
    
    report += f"\n{'='*80}\n"
    report += "EXPERIMENT SUMMARIES\n"
    report += f"{'='*80}\n\n"
    
    # Read individual experiment reports
    for exp in EXPERIMENTS:
        if results.get(exp['id'], False):
            report_file = os.path.join(exp['output_dir'], f"experiment_{exp['id']}_report.txt")
            if os.path.exists(report_file):
                with open(report_file, 'r') as f:
                    report += f.read()
                    report += "\n\n"
    
    report += f"""
{'='*80}
OVERALL RECOMMENDATIONS
{'='*80}

Based on the gap analysis findings:

1. THRESHOLD OPTIMIZATION
   - Review Experiment 1 results for optimal τ* value
   - Implement adaptive threshold based on context sensitivity

2. PATCH POSITIONING STRATEGY
   - Experiment 2 shows position-dependent effectiveness
   - Consider multi-position injection for critical deployments

3. SEMANTIC ENHANCEMENT
   - Experiment 3 identifies low-energy attack vulnerabilities
   - Enhance CNN with contextual semantic filters
   - Increase attention layer weight in ensemble decision

4. DEFENSE HARDENING
   - Experiment 4 reveals cancellation vulnerabilities
   - Deploy inoculated patches in production
   - Consider LLM-side system prompts where available

5. MONITORING & ITERATION
   - Continuously monitor for new attack patterns
   - Update safe/malicious vocabulary sets
   - Re-run gap analysis quarterly

{'='*80}
END OF CONSOLIDATED REPORT
{'='*80}
"""
    
    # Save consolidated report
    with open("SHPDE_Gap_Analysis_Consolidated_Report.txt", 'w') as f:
        f.write(report)
    
    print(report)
    print("\n✅ Consolidated report saved: SHPDE_Gap_Analysis_Consolidated_Report.txt\n")

# --- MAIN ---
def main():
    print_banner()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            # Run all experiments
            print("Running all experiments sequentially...\n")
            results = run_all_experiments()
        elif sys.argv[1] == '--exp' and len(sys.argv) > 2:
            # Run specific experiment
            exp_id = int(sys.argv[2])
            print(f"Running Experiment {exp_id} only...\n")
            success = run_specific_experiment(exp_id)
            results = {exp_id: success}
        else:
            print("Usage:")
            print("  python run_gap_analysis.py --all          # Run all experiments")
            print("  python run_gap_analysis.py --exp <1-4>    # Run specific experiment")
            print("  python run_gap_analysis.py                # Interactive mode")
            sys.exit(1)
    else:
        # Interactive mode
        print("Select execution mode:")
        print("  1. Run all experiments")
        print("  2. Run specific experiment")
        print("  3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            results = run_all_experiments()
        elif choice == '2':
            exp_id = int(input("Enter experiment ID (1-4): ").strip())
            success = run_specific_experiment(exp_id)
            results = {exp_id: success}
        else:
            print("Exiting...")
            sys.exit(0)
    
    # Generate consolidated report
    generate_consolidated_report(results)
    
    print("\n✅ Gap Analysis Complete!")
    print("📁 Check individual experiment directories for detailed results.")

if __name__ == "__main__":
    main()
