"""
Web server for multi-site Aviator predictor.

Provides REST API endpoints for predictions across all supported sites.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from functools import wraps
import logging
import json
from datetime import datetime
import threading

from aviator_predictor import MultiSiteAviatorPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global predictor instance
multi_predictor = None
update_thread = None


def initialize_predictor():
    """
    Initialize the multi-site predictor.
    """
    global multi_predictor
    logger.info("Initializing multi-site predictor...")
    multi_predictor = MultiSiteAviatorPredictor(
        sites=['betpawa', 'bongo_bongo_ug', '1xbet', 'bet365'],
        max_workers=4
    )
    logger.info("Predictor initialized successfully")


def require_prediction_ready(f):
    """
    Decorator to check if predictor is ready.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if multi_predictor is None:
            return jsonify({'error': 'Predictor not initialized'}), 503
        return f(*args, **kwargs)
    return decorated


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    """
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'version': '0.2.0'
    })


@app.route('/api/sites', methods=['GET'])
def get_supported_sites():
    """
    Get list of supported betting sites.
    """
    from aviator_predictor.site_scrapers import SiteScraperFactory
    return jsonify({
        'sites': SiteScraperFactory.get_supported_sites(),
        'active_sites': multi_predictor.sites if multi_predictor else []
    })


@app.route('/api/fetch-data', methods=['POST'])
@require_prediction_ready
def fetch_game_data():
    """
    Fetch game data from all sites.
    """
    try:
        data = request.get_json() or {}
        limit = data.get('limit', 100)
        combined = data.get('combined', True)

        logger.info(f"Fetching game data with limit={limit}")
        game_data = multi_predictor.fetch_all_game_data(limit=limit)

        if combined:
            combined_df = multi_predictor.combine_all_game_data()
            return jsonify({
                'status': 'success',
                'total_games': len(combined_df),
                'games_per_site': {
                    site: len(games) for site, games in game_data.items()
                },
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'success',
                'games_per_site': {
                    site: len(games) for site, games in game_data.items()
                },
                'timestamp': datetime.now().isoformat()
            })
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        return jsonify({'error': str(e)}), 400


@app.route('/api/train', methods=['POST'])
@require_prediction_ready
def train_models():
    """
    Train models using collected data.
    """
    try:
        data = request.get_json() or {}
        combined = data.get('combined', True)

        logger.info("Training models...")
        metrics = multi_predictor.train_all_models(combined=combined)

        return jsonify({
            'status': 'success',
            'metrics': metrics,
            'trained_sites': [site for site in multi_predictor.sites
                            if multi_predictor.predictors[site].is_trained],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        return jsonify({'error': str(e)}), 400


@app.route('/api/predict', methods=['POST'])
@require_prediction_ready
def make_predictions():
    """
    Make predictions for all sites.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No game data provided'}), 400

        logger.info("Making predictions...")
        predictions = multi_predictor.predict_all_sites(data)

        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        return jsonify({'error': str(e)}), 400


@app.route('/api/consensus', methods=['POST'])
@require_prediction_ready
def get_consensus():
    """
    Get consensus prediction across all sites.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No game data provided'}), 400

        threshold = request.args.get('threshold', 0.65, type=float)
        logger.info("Getting consensus prediction...")
        consensus = multi_predictor.get_consensus_prediction(data, threshold)

        return jsonify({
            'status': 'success',
            'consensus': consensus,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting consensus: {str(e)}")
        return jsonify({'error': str(e)}), 400


@app.route('/api/live', methods=['GET'])
@require_prediction_ready
def get_live_data():
    """
    Get live data from all sites.
    """
    try:
        logger.info("Fetching live data...")
        live_data = multi_predictor.fetch_live_data_all()

        return jsonify({
            'status': 'success',
            'live_data': live_data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error fetching live data: {str(e)}")
        return jsonify({'error': str(e)}), 400


@app.route('/api/report', methods=['GET'])
@require_prediction_ready
def get_report():
    """
    Get system report.
    """
    try:
        report = multi_predictor.generate_report()
        return jsonify({
            'status': 'success',
            'report': report,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return jsonify({'error': str(e)}), 400


@app.route('/api/predict-live', methods=['GET'])
@require_prediction_ready
def predict_live():
    """
    Get live predictions for current games across all sites.
    """
    try:
        logger.info("Generating live predictions...")
        live_data = multi_predictor.fetch_live_data_all()
        predictions = {}

        for site, game_data in live_data.items():
            site_pred = multi_predictor.predict_all_sites(game_data)
            predictions[site] = site_pred

        # Get consensus
        if live_data:
            sample_data = next(iter(live_data.values()))
            consensus = multi_predictor.get_consensus_prediction(sample_data)
        else:
            consensus = {'error': 'No live data available'}

        return jsonify({
            'status': 'success',
            'live_predictions': predictions,
            'consensus': consensus,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error generating live predictions: {str(e)}")
        return jsonify({'error': str(e)}), 400


@app.route('/', methods=['GET'])
def root():
    """
    Root endpoint with API documentation.
    """
    return jsonify({
        'name': 'Aviator Predictor API',
        'version': '0.2.0',
        'description': 'Multi-site Aviator betting predictor for Betpawa, Bongo Bongo UG, and more',
        'endpoints': {
            'health': 'GET /api/health',
            'sites': 'GET /api/sites',
            'fetch_data': 'POST /api/fetch-data',
            'train': 'POST /api/train',
            'predict': 'POST /api/predict',
            'consensus': 'POST /api/consensus',
            'live_data': 'GET /api/live',
            'live_predictions': 'GET /api/predict-live',
            'report': 'GET /api/report',
        }
    })


if __name__ == '__main__':
    initialize_predictor()
    app.run(host='0.0.0.0', port=5000, debug=True)
