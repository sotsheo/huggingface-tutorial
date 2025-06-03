from flask import Blueprint, request, jsonify
from app import db
from app.models.news import News
from app.schemas.news_schema import NewsSchema
from flask_jwt_extended import jwt_required

news_bp = Blueprint('news', __name__)
news_schema = NewsSchema()
news_list_schema = NewsSchema(many=True)

@news_bp.route('/news', methods=['GET'])
def get_news():
    news = News.query.order_by(News.created_at.desc()).all()
    return jsonify(news_list_schema.dump(news)), 200

@news_bp.route('/news/<int:news_id>', methods=['GET'])
def get_news_detail(news_id):
    news = News.query.get_or_404(news_id)
    return news_schema.dump(news), 200

@news_bp.route('/news', methods=['POST'])
@jwt_required()
def create_news():
    data = request.get_json()
    errors = news_schema.validate(data)
    if errors:
        return jsonify(errors), 400
    news = News(**data)
    db.session.add(news)
    db.session.commit()
    return news_schema.dump(news), 201

@news_bp.route('/news/<int:news_id>', methods=['PUT'])
@jwt_required()
def update_news(news_id):
    news = News.query.get_or_404(news_id)
    data = request.get_json()
    errors = news_schema.validate(data, partial=True)
    if errors:
        return jsonify(errors), 400
    for key, value in data.items():
        setattr(news, key, value)
    db.session.commit()
    return news_schema.dump(news), 200

@news_bp.route('/news/<int:news_id>', methods=['DELETE'])
@jwt_required()
def delete_news(news_id):
    news = News.query.get_or_404(news_id)
    db.session.delete(news)
    db.session.commit()
    return jsonify({'msg': 'Deleted'}), 200
