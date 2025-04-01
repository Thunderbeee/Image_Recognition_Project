import os
import random
import json
import shutil
from pathlib import Path

class ExperimentMaker:
    def __init__(self, reference_db_path, include_student=True):
        """
        Initialize with path to the reference database of labeled photos.
        
        Args:
            reference_db_path: Path to the reference database
            include_student: Whether to include photos from student directory
        """
        self.reference_db_path = Path(reference_db_path)
        self.student_path = Path("student")
        self.include_student = include_student
        self.excluded_participants = self._get_excluded_participants()
        
    def _get_excluded_participants(self):
        """Get list of excluded participants from readme files."""
        excluded = []
        readme_path = Path("data/extracted/readme.txt")
        if readme_path.exists():
            with open(readme_path, 'r') as f:
                for line in f:
                    if "withdraw" in line.lower():
                        parts = line.split("#")
                        if len(parts) > 1:
                            participant_id = parts[1].split()[0].strip()
                            excluded.append(participant_id)
        
        return excluded
    
    def create_datasets(self, templatedb_path, probes_path, 
                        max_template_individuals=100, 
                        max_probe_individuals=None,
                        images_per_template_individual=1, 
                        images_per_probe_individual=1):
        """
        Create template and probe datasets from reference database.
        
        Args:
            templatedb_path: Output path for template database
            probes_path: Output path for probes database
            max_template_individuals: Maximum number of individuals in templatedb
            max_probe_individuals: Maximum number of individuals in probes (defaults to same as template)
            images_per_template_individual: Number of images per individual in templatedb
            images_per_probe_individual: Number of images per individual in probes
        """
        if max_probe_individuals is None:
            max_probe_individuals = max_template_individuals
            
        all_individuals = []
        
        # Get individuals from main reference database
        for item in os.listdir(self.reference_db_path):
            item_path = self.reference_db_path / item
            if item_path.is_dir() and item not in self.excluded_participants:
                all_individuals.append({"id": item, "path": item_path})
        
        if len(all_individuals) < max_template_individuals:
            print(f"Warning: Only {len(all_individuals)} individuals available")
            max_template_individuals = len(all_individuals)
            max_probe_individuals = min(max_probe_individuals, max_template_individuals)
        
        random.shuffle(all_individuals)
        template_individuals = all_individuals[:max_template_individuals]
        probe_individuals = template_individuals[:max_probe_individuals]

        if self.include_student and self.student_path.exists():
            for item in os.listdir(self.student_path):
                item_path = self.student_path / item
                if item_path.is_dir() and item not in self.excluded_participants:
                    template_individuals.append({"id": item, "path": item_path})
                    probe_individuals.append({"id": item, "path": item_path})
        
        template_data = {}
        probe_data = {}
        
        for individual in template_individuals:
            individual_id = individual["id"]
            individual_path = individual["path"]
            images = [f for f in os.listdir(individual_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(images) < images_per_template_individual + images_per_probe_individual:
                print(f"Warning: Individual {individual_id} has only {len(images)} images, need at least "
                      f"{images_per_template_individual + images_per_probe_individual}")
                template_imgs = images[:min(images_per_template_individual, len(images))]
                probe_imgs = []
                if individual in probe_individuals and len(images) > images_per_template_individual:
                    probe_imgs = images[images_per_template_individual:
                                        images_per_template_individual + min(images_per_probe_individual, 
                                                                           len(images) - images_per_template_individual)]
            else:
                random.shuffle(images)
                template_imgs = images[:images_per_template_individual]
                probe_imgs = []
                if individual in probe_individuals:
                    probe_imgs = images[images_per_template_individual:
                                        images_per_template_individual + images_per_probe_individual]
            
            template_data[individual_id] = [str(individual_path / img) for img in template_imgs]
            
            if individual in probe_individuals and probe_imgs:
                probe_data[individual_id] = [str(individual_path / img) for img in probe_imgs]
        
        os.makedirs(os.path.dirname(templatedb_path), exist_ok=True)
        os.makedirs(os.path.dirname(probes_path), exist_ok=True)
        
        with open(templatedb_path, 'w') as f:
            json.dump(template_data, f, indent=2)
            
        with open(probes_path, 'w') as f:
            json.dump(probe_data, f, indent=2)
            
        print(f"Created template database with {len(template_data)} individuals")
        print(f"Created probe dataset with {len(probe_data)} individuals")
        
        return template_data, probe_data

if __name__ == "__main__":
    maker = ExperimentMaker("data/extracted", include_student=True)
    template_data, probe_data = maker.create_datasets(
        "data/experiment/templatedb.json",
        "data/experiment/probes.json",
        max_template_individuals=50,
        images_per_template_individual=1,
        images_per_probe_individual=1
    ) 