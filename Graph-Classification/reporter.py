import json
import os
import numpy as np
from datetime import datetime

class GraphClassificationReporter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.results_file = os.path.join(output_dir, "graph_classification_results.json")
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
        key = f"{run_result['dataset']},{run_result['model']},{run_result['prune_type']},{run_result['drop_rate']}"
        
        if key not in self.results:
            self.results[key] = {
                'accuracies': [],
                'execution_times': [],
                'config': {
                    'dataset': run_result['dataset'],
                    'model': run_result['model'],
                    'prune_type': run_result['prune_type'],
                    'drop_rate': run_result['drop_rate']
                }
            }
        
        self.results[key]['accuracies'].append(run_result['accuracy'])
        self.results[key]['execution_times'].append(run_result['execution_time'])
        self.save_results()
        
        print(f"Run {run_result['run'] + 1}: Accuracy = {run_result['accuracy'] * 100:.2f}%")
    
    def report_summary(self, results, execution_times):
        if not results:
            return
        
        # Group results by configuration
        configs = {}
        for result in results:
            key = (result['dataset'], result['model'], result['prune_type'], result['drop_rate'])
            if key not in configs:
                configs[key] = []
            configs[key].append(result['accuracy'])
        
        print("\n" + "="*80)
        print("GRAPH CLASSIFICATION SUMMARY REPORT")
        print("="*80)
        
        for config, accuracies in configs.items():
            dataset, model, prune_type, drop_rate = config
            
            avg_accuracy = np.mean(accuracies) * 100
            std_accuracy = np.std(accuracies) * 100
            avg_time = np.mean(execution_times)
            
            print(f"\n{dataset}, {model}, {prune_type}, DropRate={drop_rate}:")
            print(f"  Average Accuracy: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%")
            print(f"  Average Time per Run: {avg_time:.2f}s")
            print(f"  Individual Accuracies: {[f'{acc*100:.2f}%' for acc in accuracies]}")
        
        print(f"\nResults saved to: {self.results_file}")