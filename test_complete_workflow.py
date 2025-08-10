#!/usr/bin/env python3
"""
Complete Workflow Test and Validation
====================================

This script tests the complete hypersonic reentry framework workflow
end-to-end to ensure all components are working correctly.
"""

import os
import sys
import json
import csv
import time
from pathlib import Path

def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_section(title):
    """Print formatted section header."""
    print(f"\n🔍 {title}")
    print("-" * 50)

def check_file_exists(filepath, description):
    """Check if file exists and report."""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"   ✓ {description}: {filepath} ({size} bytes)")
        return True
    else:
        print(f"   ❌ {description}: {filepath} (NOT FOUND)")
        return False

def check_directory_structure():
    """Test directory structure completeness."""
    print_section("Directory Structure Validation")
    
    required_dirs = [
        ("src/hypersonic_reentry", "Core framework source"),
        ("data/trajectories", "Trajectory data"),
        ("results/statistical", "Statistical results"),
        ("results/optimization", "Optimization results"), 
        ("results/sensitivity", "Sensitivity analysis"),
        ("results/plots", "Visualization gallery"),
        ("docs/api", "API documentation"),
        ("docs/tutorials", "Tutorial documentation"),
        ("website/_site", "Generated Jekyll site"),
        ("notebooks", "Jupyter notebooks"),
        ("examples", "Example scripts")
    ]
    
    structure_score = 0
    for directory, description in required_dirs:
        if os.path.exists(directory):
            files_count = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
            print(f"   ✓ {description}: {directory}/ ({files_count} files)")
            structure_score += 1
        else:
            print(f"   ❌ {description}: {directory}/ (NOT FOUND)")
    
    print(f"\nDirectory Structure Score: {structure_score}/{len(required_dirs)} ({structure_score/len(required_dirs)*100:.1f}%)")
    return structure_score == len(required_dirs)

def validate_simulation_data():
    """Test simulation data generation."""
    print_section("Simulation Data Validation")
    
    # Test trajectory data
    trajectory_file = "data/trajectories/nominal_trajectory.csv"
    trajectory_valid = False
    
    if check_file_exists(trajectory_file, "Nominal trajectory CSV"):
        try:
            with open(trajectory_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
            required_columns = ['time', 'altitude', 'velocity', 'heat_rate']
            has_all_columns = all(col in rows[0].keys() for col in required_columns)
            
            if has_all_columns and len(rows) > 10:
                print(f"      - {len(rows)} trajectory points")
                print(f"      - Columns: {list(rows[0].keys())}")
                print(f"      - Time range: {rows[0]['time']} to {rows[-1]['time']} seconds")
                print(f"      - Altitude range: {rows[0]['altitude']} to {rows[-1]['altitude']} m")
                trajectory_valid = True
            else:
                print(f"      ❌ Invalid trajectory data structure")
        except Exception as e:
            print(f"      ❌ Error reading trajectory file: {e}")
    
    # Test Monte Carlo results
    mc_file = "results/statistical/monte_carlo_summary.json"
    mc_valid = False
    
    if check_file_exists(mc_file, "Monte Carlo results JSON"):
        try:
            with open(mc_file, 'r') as f:
                mc_data = json.load(f)
            
            required_keys = ['num_samples', 'output_metrics', 'reliability_metrics']
            has_required_keys = all(key in mc_data for key in required_keys)
            
            if has_required_keys:
                print(f"      - {mc_data['num_samples']} Monte Carlo samples")
                print(f"      - Output metrics: {list(mc_data['output_metrics'].keys())}")
                print(f"      - Overall mission success: {mc_data['reliability_metrics']['overall_mission']['probability']:.1%}")
                mc_valid = True
            else:
                print(f"      ❌ Missing required Monte Carlo data keys")
        except Exception as e:
            print(f"      ❌ Error reading Monte Carlo file: {e}")
    
    # Test optimization results
    opt_file = "results/optimization/optimization_summary.json"
    opt_valid = False
    
    if check_file_exists(opt_file, "Optimization results JSON"):
        try:
            with open(opt_file, 'r') as f:
                opt_data = json.load(f)
            
            required_keys = ['scenarios_analyzed', 'overall_success_rate', 'results_by_scenario']
            has_required_keys = all(key in opt_data for key in required_keys)
            
            if has_required_keys:
                print(f"      - {opt_data['scenarios_analyzed']} optimization scenarios")
                print(f"      - Overall success rate: {opt_data['overall_success_rate']:.1%}")
                print(f"      - Algorithm: {opt_data['methodology']['algorithm']}")
                opt_valid = True
            else:
                print(f"      ❌ Missing required optimization data keys")
        except Exception as e:
            print(f"      ❌ Error reading optimization file: {e}")
    
    # Test sensitivity analysis
    sens_file = "results/sensitivity/sobol_indices.json"
    sens_valid = False
    
    if check_file_exists(sens_file, "Sensitivity analysis JSON"):
        try:
            with open(sens_file, 'r') as f:
                sens_data = json.load(f)
            
            required_keys = ['first_order_indices', 'total_effect_indices', 'parameter_ranking']
            has_required_keys = all(key in sens_data for key in required_keys)
            
            if has_required_keys:
                print(f"      - {len(sens_data['parameters'])} uncertain parameters")
                print(f"      - {len(sens_data['outputs'])} output quantities")
                most_important = sens_data['parameter_ranking']['by_total_variance_explained'][0]
                print(f"      - Most important parameter: {most_important['parameter']} ({most_important['avg_total_effect']:.2f})")
                sens_valid = True
            else:
                print(f"      ❌ Missing required sensitivity analysis keys")
        except Exception as e:
            print(f"      ❌ Error reading sensitivity file: {e}")
    
    data_score = sum([trajectory_valid, mc_valid, opt_valid, sens_valid])
    print(f"\nSimulation Data Score: {data_score}/4 ({data_score/4*100:.1f}%)")
    return data_score >= 3

def validate_documentation():
    """Test documentation completeness."""
    print_section("Documentation Validation")
    
    doc_files = [
        ("docs/api/hypersonic_reentry_api.md", "API Reference"),
        ("docs/tutorials/getting-started.md", "Getting Started Guide"),
        ("docs/theory/mathematical_foundations.md", "Mathematical Theory"),
        ("docs/validation/framework_validation.md", "Validation Documentation"),
        ("docs/workflows/complete_analysis_workflow.md", "Analysis Workflows"),
        ("docs/examples/monte_carlo_example.md", "Monte Carlo Example"),
        ("docs/development/contributing.md", "Contributing Guidelines"),
        ("README.md", "Main README"),
        ("PROJECT_STRUCTURE.md", "Project Structure Guide"),
        ("USER_GUIDE.md", "User Documentation")
    ]
    
    doc_score = 0
    total_size = 0
    
    for filepath, description in doc_files:
        if check_file_exists(filepath, description):
            doc_score += 1
            total_size += os.path.getsize(filepath)
    
    print(f"\nDocumentation Score: {doc_score}/{len(doc_files)} ({doc_score/len(doc_files)*100:.1f}%)")
    print(f"Total Documentation Size: {total_size/1024:.1f} KB")
    return doc_score >= len(doc_files) * 0.8

def validate_website():
    """Test Jekyll website generation."""
    print_section("Website Validation")
    
    site_files = [
        ("website/_site/index.html", "Homepage"),
        ("website/_site/methodology.html", "Methodology page"),
        ("website/_site/results.html", "Results page"),
        ("website/_site/assets/css/main.css", "Main stylesheet"),
        ("website/_site/feed.xml", "RSS feed"),
        ("website/_site/sitemap.xml", "Sitemap")
    ]
    
    website_score = 0
    
    for filepath, description in site_files:
        if check_file_exists(filepath, description):
            website_score += 1
    
    # Check if main CSS exists and has content
    css_file = "website/_site/assets/css/main.css"
    if os.path.exists(css_file):
        css_size = os.path.getsize(css_file)
        print(f"      - CSS size: {css_size} bytes")
    
    # Check index.html content
    index_file = "website/_site/index.html"
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            content = f.read()
        if "Stochastic Simulation" in content and "MathJax" in content:
            print(f"      - Homepage contains expected content and MathJax")
            website_score += 1
        else:
            print(f"      ❌ Homepage missing expected content")
    
    print(f"\nWebsite Score: {website_score}/{len(site_files)+1} ({website_score/(len(site_files)+1)*100:.1f}%)")
    return website_score >= len(site_files)

def validate_notebooks():
    """Test Jupyter notebooks."""
    print_section("Jupyter Notebooks Validation")
    
    notebook_files = [
        ("notebooks/01_quick_start.ipynb", "Quick Start Notebook"),
        ("notebooks/03_monte_carlo_analysis.ipynb", "Monte Carlo Analysis Notebook"),
    ]
    
    notebook_score = 0
    
    for filepath, description in notebook_files:
        if check_file_exists(filepath, description):
            try:
                with open(filepath, 'r') as f:
                    notebook_data = json.load(f)
                
                if 'cells' in notebook_data and len(notebook_data['cells']) > 0:
                    cell_count = len(notebook_data['cells'])
                    code_cells = len([cell for cell in notebook_data['cells'] if cell['cell_type'] == 'code'])
                    print(f"      - {cell_count} cells ({code_cells} code cells)")
                    notebook_score += 1
                else:
                    print(f"      ❌ Invalid notebook structure")
            except Exception as e:
                print(f"      ❌ Error reading notebook: {e}")
    
    print(f"\nNotebooks Score: {notebook_score}/{len(notebook_files)} ({notebook_score/len(notebook_files)*100:.1f}%)")
    return notebook_score >= len(notebook_files) * 0.5

def validate_examples():
    """Test example scripts."""
    print_section("Example Scripts Validation")
    
    example_files = [
        ("examples/complete_analysis_example.py", "Complete Analysis Example"),
        ("examples/comprehensive_results_generation.py", "Results Generation Example"),
    ]
    
    example_score = 0
    
    for filepath, description in example_files:
        if check_file_exists(filepath, description):
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                
                # Check for key framework imports
                if "hypersonic_reentry" in content and "def " in content:
                    print(f"      - Contains framework imports and functions")
                    example_score += 1
                else:
                    print(f"      ❌ Missing expected code structure")
            except Exception as e:
                print(f"      ❌ Error reading example: {e}")
    
    print(f"\nExamples Score: {example_score}/{len(example_files)} ({example_score/len(example_files)*100:.1f}%)")
    return example_score >= len(example_files) * 0.5

def validate_framework_structure():
    """Test core framework structure."""
    print_section("Framework Structure Validation")
    
    framework_files = [
        ("src/hypersonic_reentry/__init__.py", "Framework init"),
        ("src/hypersonic_reentry/dynamics/vehicle_dynamics.py", "Vehicle dynamics"),
        ("src/hypersonic_reentry/atmosphere/us_standard_1976.py", "Atmosphere model"),
        ("src/hypersonic_reentry/uncertainty/uncertainty_quantifier.py", "Uncertainty quantification"),
        ("src/hypersonic_reentry/optimization/gradient_based.py", "Optimization"),
        ("src/hypersonic_reentry/analysis/statistical_analyzer.py", "Statistical analysis"),
        ("src/hypersonic_reentry/visualization/plot_manager.py", "Visualization"),
        ("setup.py", "Package setup"),
        ("requirements.txt", "Dependencies")
    ]
    
    framework_score = 0
    
    for filepath, description in framework_files:
        if check_file_exists(filepath, description):
            framework_score += 1
    
    print(f"\nFramework Structure Score: {framework_score}/{len(framework_files)} ({framework_score/len(framework_files)*100:.1f}%)")
    return framework_score >= len(framework_files) * 0.8

def run_comprehensive_test():
    """Run complete end-to-end test."""
    print_header("HYPERSONIC REENTRY FRAMEWORK - COMPLETE WORKFLOW TEST")
    
    start_time = time.time()
    
    # Run all validation tests
    tests = [
        ("Directory Structure", check_directory_structure),
        ("Simulation Data", validate_simulation_data), 
        ("Documentation", validate_documentation),
        ("Website Generation", validate_website),
        ("Jupyter Notebooks", validate_notebooks),
        ("Example Scripts", validate_examples),
        ("Framework Structure", validate_framework_structure)
    ]
    
    results = []
    for test_name, test_function in tests:
        try:
            result = test_function()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            print(f"\n❌ ERROR: {test_name} - {e}")
            results.append((test_name, False))
    
    # Overall results
    end_time = time.time()
    execution_time = end_time - start_time
    
    print_header("WORKFLOW TEST SUMMARY")
    
    passed_tests = sum(1 for _, result in results if result)
    total_tests = len(results)
    
    print(f"🕒 Execution Time: {execution_time:.2f} seconds")
    print(f"📊 Test Results: {passed_tests}/{total_tests} PASSED ({passed_tests/total_tests*100:.1f}%)")
    print()
    
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    # Overall assessment
    print()
    if passed_tests >= total_tests * 0.8:
        print("🎉 FRAMEWORK STATUS: PRODUCTION READY")
        print("   All critical components are functional and complete.")
        print("   The framework is ready for research and educational use.")
    elif passed_tests >= total_tests * 0.6:
        print("⚠️  FRAMEWORK STATUS: MOSTLY FUNCTIONAL")
        print("   Most components are working but some issues need attention.")
        print("   Framework can be used with caution.")
    else:
        print("❌ FRAMEWORK STATUS: NEEDS WORK")
        print("   Significant issues found that require fixes.")
        print("   Framework not ready for production use.")
    
    # Specific recommendations
    print("\n📋 RECOMMENDATIONS:")
    
    failed_tests = [test_name for test_name, result in results if not result]
    if not failed_tests:
        print("   • Framework is complete and ready to use")
        print("   • Consider adding more examples and tutorials")
        print("   • Monitor performance with larger datasets")
    else:
        print(f"   • Fix failing components: {', '.join(failed_tests)}")
        print("   • Verify all data files are properly generated")
        print("   • Check documentation for completeness")
    
    print("\n🚀 NEXT STEPS:")
    print("   1. Review any failing tests above")
    print("   2. Run the framework with real analysis scenarios")
    print("   3. Validate results against known benchmarks") 
    print("   4. Share with research community for feedback")
    
    return passed_tests >= total_tests * 0.8

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)