# Facial-Skin/backend/convert.py
import os
from flask import request, jsonify
from flask_restful import Resource
import traceback
from models.recommender.rec import recs_essentials, makeup_recommendation  # ADD THIS IMPORT

model_path = './models/skin_model/'
print(os.listdir(model_path))

class Recommendation(Resource):
    def post(self):
        try:
            data = request.json
            print("\n=== Received recommendation request with data: ===")
            print(f"Skin Type: {data['skin_type']}")
            print(f"Skin Tone: {data['tone']}")
            print("Features:")
            
            # Validate feature structure
            if 'features' not in data:
                return {"error": "Missing features data"}, 400
                
            for feature, value in data['features'].items():
                print(f" - {feature}: {value}")

            # Convert features to proper format with error handling
            try:
                feature_vector = [int(data['features'][f]) for f in [
                    'normal', 'dry', 'oily', 'combination', 'acne', 'sensitive',
                    'fine lines', 'wrinkles', 'redness', 'dull', 'pore',
                    'pigmentation', 'blackheads', 'whiteheads', 'blemishes',
                    'dark circles', 'eye bags', 'dark spots'
                ]]
            except KeyError as e:
                return {"error": f"Missing feature: {str(e)}"}, 400

            print("\n=== Starting recommendation engine ===")
            print(f"Feature vector: {feature_vector}")

            # Get recommendations with null checks
            print("\nGenerating general recommendations...")
            general = recs_essentials(feature_vector, None)
            if not general:
                print("Warning: No general recommendations generated")

            print("\nGenerating makeup recommendations...")
            makeup = makeup_recommendation(
                data['tone'], 
                data['skin_type'].lower()
            )
            if not makeup:
                print("Warning: No makeup recommendations generated")

            print("\n=== Final recommendations ===")
            print(f"General recommendations count: {sum(len(v) for v in general.values())}")
            print(f"Makeup recommendations count: {len(makeup)}")
            
            return {
                "general_recommendations": general,
                "makeup_recommendations": makeup
            }, 200
            
        except Exception as e:
            traceback.print_exc()
            return {"error": str(e)}, 500