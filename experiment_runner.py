import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from alpha_prototype import AlphaPrototype

class ExperimentRunner:
    def __init__(self, templatedb_path, probes_path, output_dir="results"):
        """
        Initialize experiment runner.
        
        Args:
            templatedb_path: Path to template database JSON
            probes_path: Path to probes dataset JSON
            output_dir: Directory to save experiment results
            use_gpu: Whether to attempt using GPU for model inference
        """
        self.templatedb_path = templatedb_path
        self.probes_path = probes_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        with open(probes_path, 'r') as f:
            self.probe_data = json.load(f)
            
        self.results = []
    
    def run(self, model_name="VGG-Face", distance_metric="cosine", threshold=None):
        """
        Run experiment with specified parameters.
        
        Args:
            model_name: Face recognition model to use
            distance_metric: Distance metric for face matching
            threshold: Optional threshold for accepting matches
            
        Returns:
            DataFrame with experiment results
        """
        print(f"Running experiment with model={model_name}, metric={distance_metric}")
        
        prototype = AlphaPrototype(self.templatedb_path, model_name, distance_metric)
        
        total_probes = sum(len(imgs) for imgs in self.probe_data.values())
        processed = 0
        
        for true_person_id, image_paths in tqdm(self.probe_data.items(), desc="Processing individuals"):
            for img_path in image_paths:
                try:
                    result = prototype.identify(img_path, threshold)
                    
                    self.results.append({
                        "probe_image": img_path,
                        "true_person_id": true_person_id,
                        "predicted_person_id": result["person_id"],
                        "distance": result["distance"],
                        "match_accepted": result["match_accepted"],
                        "error": None
                    })
                except Exception as e:
                    print(f"Error processing image {img_path}: {str(e)}")
                    self.results.append({
                        "probe_image": img_path,
                        "true_person_id": true_person_id,
                        "predicted_person_id": None,
                        "distance": None,
                        "match_accepted": False,
                        "error": str(e)
                    })
                
                processed += 1
                if processed % 10 == 0:
                    print(f"Processed {processed}/{total_probes} probe images")
        
        results_df = pd.DataFrame(self.results)
        
        self._calculate_metrics(results_df)
        
        results_path = self.output_dir / f"results_{model_name}_{distance_metric}.csv"
        results_df.to_csv(results_path, index=False)
        
        return results_df
    
    def _calculate_metrics(self, results_df):
        """Calculate and print performance metrics."""
        total = len(results_df)
        
        correct = results_df["true_person_id"] == results_df["predicted_person_id"]
        accuracy = correct.mean() * 100
        
        accepted = results_df["match_accepted"] == True
        precision = (correct & accepted).sum() / accepted.sum() * 100 if accepted.sum() > 0 else 0
        
        rejection_rate = (~accepted).mean() * 100
        
        print("\n===== EXPERIMENT RESULTS =====")
        print(f"Total probe images: {total}")
        print(f"Overall accuracy: {accuracy:.2f}%")
        print(f"Precision of accepted matches: {precision:.2f}%")
        print(f"Match rejection rate: {rejection_rate:.2f}%")
        
        true_positives = (correct & accepted).sum()
        false_positives = (~correct & accepted).sum()
        true_negatives = (~correct & ~accepted).sum()
        false_negatives = (correct & ~accepted).sum()
        
        print("\n----- Confusion Matrix -----")
        print(f"True Positives: {true_positives} ({true_positives/total*100:.2f}%)")
        print(f"False Positives: {false_positives} ({false_positives/total*100:.2f}%)")
        print(f"True Negatives: {true_negatives} ({true_negatives/total*100:.2f}%)")
        print(f"False Negatives: {false_negatives} ({false_negatives/total*100:.2f}%)")
        
        by_person = results_df.groupby("true_person_id").apply(
            lambda x: pd.Series({
                "accuracy": (x["true_person_id"] == x["predicted_person_id"]).mean() * 100,
                "count": len(x)
            })
        )
        
        print("\n----- Performance by Individual -----")
        print(f"Best recognized individual: {by_person['accuracy'].idxmax()} ({by_person['accuracy'].max():.2f}%)")
        print(f"Worst recognized individual: {by_person['accuracy'].idxmin()} ({by_person['accuracy'].min():.2f}%)")
        print(f"Average per-individual accuracy: {by_person['accuracy'].mean():.2f}%")

if __name__ == "__main__":
    runner = ExperimentRunner(
        "data/experiment/templatedb.json",
        "data/experiment/probes.json",
        "data/experiment/results",
    )
    
    results = runner.run()
    
    # Optionally, run with different models or parameters
    # results2 = runner.run(model_name="Facenet", distance_metric="euclidean", threshold=0.6) 