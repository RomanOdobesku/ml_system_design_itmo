from locust import HttpUser, between, task


# Класс для симуляции пользователя
class PredictionTestUser(HttpUser):
    wait_time = between(1, 2)  # Задержка между запросами

    # Задача для отправки запросов
    @task
    def predict(self):
        payload = {
            "data": [
                {
                    "user_id": 120738,
                    "adv_campaign_id": 195,
                    "platform_id": 2,
                    "adv_creative_id": 3267,
                    "event_date": "2024-09-21T00:00:00",
                    "banner_code": 8,
                    "is_main": True,
                    "dayofweek": 5,
                    "is_weekend": 1,
                    "end_date": "2024-09-27T00:00:00",
                    "days_to_campaign_end": 6,
                    "is_campaign_early": False,
                    "user_click_rate": 0.01418,
                    "user_impressions_count": 141,
                    "user_campaign_diversity": 67,
                    "campaign_ctr": 0.007122,
                    "campaign_impressions": 13760,
                    "campaign_budget_per_day": 576.5,
                    "logcat_id": 65,
                    "creative_click_rate": 0.007122,
                    "creative_impressions": 13760,
                    "banner_click_rate": 0.005474,
                    "banner_impressions": 74296252,
                    "platform_ctr": 0.006027,
                    "microcat_popularity": 551,
                    "parent_microcat_count": 93,
                    "user_campaign_interaction_rate": 0.0,
                },
                {
                    "user_id": 1393763,
                    "adv_campaign_id": 2632,
                    "platform_id": 2,
                    "adv_creative_id": 3316,
                    "event_date": "2024-09-21T00:00:00",
                    "banner_code": 8,
                    "is_main": True,
                    "dayofweek": 5,
                    "is_weekend": 1,
                    "end_date": "2024-09-29T00:00:00",
                    "days_to_campaign_end": 8,
                    "is_campaign_early": False,
                    "user_click_rate": 0.01563,
                    "user_impressions_count": 64,
                    "user_campaign_diversity": 43,
                    "campaign_ctr": 0.005787,
                    "campaign_impressions": 70334,
                    "campaign_budget_per_day": 2662.0,
                    "logcat_id": 18,
                    "creative_click_rate": 0.005787,
                    "creative_impressions": 70334,
                    "banner_click_rate": 0.005474,
                    "banner_impressions": 74296252,
                    "platform_ctr": 0.006027,
                    "microcat_popularity": 6260,
                    "parent_microcat_count": 2364,
                    "user_campaign_interaction_rate": 1.0,
                },
            ]
        }
        # Отправляем POST-запрос на эндпоинт /predict
        self.client.post("/predict", json=payload)
