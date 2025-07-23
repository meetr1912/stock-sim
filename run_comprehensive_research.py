"""
Execute Comprehensive PEAD Research Pipeline
"""

from comprehensive_pead_research import run_comprehensive_research
import sys

def main():
    print("🚀 Starting Comprehensive PEAD Research Pipeline...")
    print("Phases: Advanced Strategy → Expanded Data → ML Integration → Projections")
    print("Estimated time: 10-20 minutes")
    print("="*80)
    
    try:
        # Run the complete research pipeline
        results = run_comprehensive_research()
        
        if results.get('status') == 'failed':
            print(f"❌ Research failed at phase {results.get('phase', 'unknown')}")
            return 1
        
        print("\n🎉 COMPREHENSIVE RESEARCH COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Print key findings
        if 'summary' in results:
            print("\n📋 EXECUTIVE SUMMARY:")
            print(results['summary'])
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during research: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 