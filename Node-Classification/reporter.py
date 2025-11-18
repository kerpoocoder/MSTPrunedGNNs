import json
import os
import numpy as np
from datetime import datetime

class Reporter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.results_file = os.path.join(output_dir, "results.json")
        self.load_existing_results()
    
    def load_existing_results(self):
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
        else:
            self.results = {}
    
    def save_results(self):
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=4)
    
    def report_run(self, run_result):
        key = f"{run_result['dataset']},{run_result['model']},{run_result['prune_type']},{run_result['hidden_channels']}"
        
        for metric in ['accuracy', 'macro_f1', 'micro_f1']:
            if metric in run_result:
                metric_key = f"{key},{metric}"
                if metric_key not in self.results:
                    self.results[metric_key] = []
                self.results[metric_key].append(run_result[metric])
        
        self.save_results()
        
        print(f"Run {run_result['run'] + 1}: {run_result}")
    
    def report_summary(self, results):
        if not results:
            return
        
        # Group results by configuration
        configs = {}
        for result in results:
            key = (result['dataset'], result['model'], result['prune_type'], result['hidden_channels'])
            if key not in configs:
                configs[key] = []
            configs[key].append(result)
        
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)
        
        for config, runs in configs.items():
            dataset, model, prune_type, hidden_channels = config
            
            accuracies = [r.get('accuracy', 0) for r in runs]
            macro_f1s = [r.get('macro_f1', 0) for r in runs]
            micro_f1s = [r.get('micro_f1', 0) for r in runs]
            
            print(f"\n{dataset}, {model}, {prune_type}, hidden={hidden_channels}:")
            if accuracies and accuracies[0] != 0:
                print(f"  Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
            if macro_f1s and macro_f1s[0] != 0:
                print(f"  Macro F1: {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
            if micro_f1s and micro_f1s[0] != 0:
                print(f"  Micro F1: {np.mean(micro_f1s):.4f} ± {np.std(micro_f1s):.4f}")
        
        print(f"\nResults saved to: {self.results_file}")