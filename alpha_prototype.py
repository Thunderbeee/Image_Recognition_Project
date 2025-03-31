import json
from deepface import DeepFace
import numpy as np
from tqdm import tqdm


class AlphaPrototype:
    def __init__(self, templatedb_path, model_name="VGG-Face", distance_metric="cosine"):
        """
        Initialize the facial recognition system.
        
        Args:
            templatedb_path: Path to the template database JSON file
            model_name: Face recognition model to use (VGG-Face, Facenet, etc.)
            distance_metric: Distance metric for face matching (cosine, euclidean, etc.)
        """
        
        self.templatedb_path = templatedb_path
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.template_db = {}
        self.template_embeddings = {}
        self.names = {}
        
        self._load_templates()
        
    def _load_templates(self):
        """Load and process all templates from the database."""
        print(f"Loading templates from {self.templatedb_path}...")
        
        with open(self.templatedb_path, 'r') as f:
            template_data = json.load(f)

        for person_id, image_paths in tqdm(template_data.items()):
            self.names[person_id] = person_id
            self.template_db[person_id] = image_paths
            
            for i, img_path in enumerate(image_paths):
                embedding_obj = DeepFace.represent(
                    img_path=img_path, 
                    model_name=self.model_name, 
                )
                
                embedding_key = f"{person_id}_{i}"
                self.template_embeddings[embedding_key] = {
                    "embedding": embedding_obj[0]["embedding"],
                    "person_id": person_id,
                    "image_path": img_path
                }
                
        
        print(f"Loaded {len(self.template_embeddings)} face templates for {len(self.template_db)} individuals")
    
    def identify(self, query_image_path, threshold=None):
        """
        Identify a person in the query image.
        
        Args:
            query_image_path: Path to the query image
            threshold: Optional distance threshold for accepting a match
            
        Returns:
            Dictionary containing the match result with person_id, confidence, etc.
        """
        try:
            print(f"Processing query image: {query_image_path}")
            
            query_embedding_obj = DeepFace.represent(
                query_image_path, 
                model_name=self.model_name, 
            )
            query_embedding = query_embedding_obj[0]["embedding"]
            
            best_match = None
            best_distance = float('inf')
            
            for embedding_key, template in self.template_embeddings.items():
                template_embedding = template["embedding"]
                
                if self.distance_metric == "cosine":
                    distance = self._cosine_distance(query_embedding, template_embedding)
                elif self.distance_metric == "euclidean":
                    distance = self._euclidean_distance(query_embedding, template_embedding)
                else:
                    raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = template["person_id"]
            
            match_accepted = True
            if threshold is not None and best_distance > threshold:
                match_accepted = False
            
            return {
                "person_id": best_match if match_accepted else None,
                "name": self.names.get(best_match, "Unknown") if match_accepted else "No match",
                "distance": best_distance,
                "match_accepted": match_accepted
            }
            
        except Exception as e:
            print(f"Error identifying person in {query_image_path}: {str(e)}")
            return {"person_id": None, "name": "Error", "distance": None, "match_accepted": False}
    
    def _cosine_distance(self, embedding1, embedding2):
        a = np.array(embedding1)
        b = np.array(embedding2)
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _euclidean_distance(self, embedding1, embedding2):
        a = np.array(embedding1)
        b = np.array(embedding2)
        return np.linalg.norm(a - b)

if __name__ == "__main__":
    # change to your own path there
    # and make sure the path is correct
    # sanity check
    prototype = AlphaPrototype("data/experiment/templatedb.json")
    result = prototype.identify("/home/azureuser/mingyuan/alpha-prototype/data/extracted/52/TD_RGB_E_5.jpg")
    print(f"Identified as: {result['name']} (Distance: {result['distance']})")