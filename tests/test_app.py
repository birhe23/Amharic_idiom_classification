import unittest
from app import app


class LoginUITestCase(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_login_page_loads(self):
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Login', response.data)

    def test_valid_login_redirects_to_dashboard(self):
        response = self.client.post('/login', data={'username': 'admin', 'password': 'admin123'}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Dashboard', response.data)

    def test_dashboard_shows_data_summary(self):
        response = self.client.post('/login', data={'username': 'admin', 'password': 'admin123'}, follow_redirects=True)
        self.assertIn(b'Dataset Summary', response.data)

    def test_dashboard_has_table_and_predict_section(self):
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'All Idioms', response.data)
        self.assertIn(b'Predict a New Idiom', response.data)

    def test_dashboard_filtering_works(self):
        response = self.client.get('/dashboard?search=%E1%88%80%E1%88%9E%E1%89%B3&label=1')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'\xe1\x88\x80\xe1\x88\x9e\xe1\x89\xb3', response.data)

    def test_dashboard_has_pagination_controls(self):
        response = self.client.get('/dashboard?page=2')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Previous', response.data)
        self.assertIn(b'Next', response.data)

    def test_predict_route_uses_model_prediction(self):
        response = self.client.post('/predict', data={'idiom': 'ሀሞታም'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Prediction:', response.data)
        self.assertIn(b'Confidence', response.data)

    def test_dashboard_supports_richer_table_controls(self):
        response = self.client.get('/dashboard?search=ሀ&sort=label')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Search', response.data)
        self.assertIn(b'Sort by', response.data)


if __name__ == '__main__':
    unittest.main()
