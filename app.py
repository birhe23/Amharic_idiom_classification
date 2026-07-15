import csv
import io
import json
import os
from pathlib import Path

from flask import Flask, make_response, render_template_string, request, send_file, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'labeled_idiom.csv')
MODEL_RESULTS_PATH = os.path.join(BASE_DIR, 'model_results.json')
MODEL_CACHE_PATH = os.path.join(BASE_DIR, 'classifier_model.joblib')

LOGIN_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Login</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {
      font-family: 'Noto Sans Ethiopic', 'Noto Sans', 'Segoe UI', Arial, sans-serif;
      background: linear-gradient(135deg, #eef4ff 0%, #f8fbff 100%);
    }
    .auth-card {
      max-width: 430px;
      width: 100%;
      border: 0;
      border-radius: 1.25rem;
      box-shadow: 0 18px 45px rgba(37, 99, 235, 0.16);
    }
  </style>
</head>
<body class="min-vh-100 d-flex align-items-center justify-content-center p-3">
  <div class="card auth-card p-4 p-md-5">
    <div class="text-center mb-4">
      <h2 class="fw-bold mb-2">Login</h2>
      <p class="text-muted mb-0">እባክዎ ወደ ስርዓቱ ይግቡ</p>
    </div>
    <form method="post" action="{{ url_for('login') }}">
      <div class="mb-3">
        <label class="form-label">Username</label>
        <input type="text" class="form-control" name="username" placeholder="Username" required>
      </div>
      <div class="mb-3">
        <label class="form-label">Password</label>
        <input type="password" class="form-control" name="password" placeholder="Password" required>
      </div>
      <button type="submit" class="btn btn-primary w-100">Sign In</button>
    </form>
    {% if error %}<div class="alert alert-danger mt-3 mb-0">{{ error }}</div>{% endif %}
  </div>
</body>
</html>
"""

DASHBOARD_HTML = """
<!doctype html>
<html lang="am">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dashboard</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <style>
    body {
      font-family: 'Noto Sans Ethiopic', 'Noto Sans', 'Segoe UI', Arial, sans-serif;
      background: #f5f8ff;
    }
    .sidebar {
      position: sticky;
      top: 0;
      height: 100vh;
      overflow-y: auto;
      background: linear-gradient(180deg, #14213d 0%, #1f3b72 100%);
      /* subtle right shadow to separate from main content */
      box-shadow: 6px 0 18px rgba(9, 30, 66, 0.08);
      z-index: 10;
    }
    .sidebar .nav {
      max-height: calc(100vh - 120px);
    }
    .nav-link.active {
      background: rgba(255,255,255,0.15);
      border-radius: 0.5rem;
    }
    .main-content {
      max-height: 100vh;
      overflow-y: auto;
      /* subtle divider on the left for clearer split */
      border-left: 1px solid rgba(15, 23, 42, 0.06);
      padding-left: 2rem;
    }
  </style>
</head>
<body>
  <div class="container-fluid">
    <div class="row min-vh-100">
      <aside class="sidebar col-md-3 col-lg-2 text-white p-0">
        <div class="p-4 border-bottom border-secondary">
          <h4 class="fw-bold mb-1">Dashboard</h4>
          <p class="small mb-0 text-light opacity-75">አማርኛ የእንቅስቃሴ ፓነል</p>
        </div>
        <nav class="nav flex-column p-3">
          <a class="nav-link text-white active mb-2" href="#overview">Overview</a>
          <a class="nav-link text-white mb-2" href="#dataset">Dataset</a>
          <a class="nav-link text-white mb-2" href="#models">Models</a>
          <a class="nav-link text-white mb-2" href="#idioms">Idioms</a>
          <a class="nav-link text-white" href="#predict">Predict</a>
        </nav>
      </aside>
      <main class="main-content col-md-9 col-lg-10 p-4 bg-light">
        <div class="d-flex flex-column flex-md-row justify-content-between align-items-md-center mb-4">
          <div>
            <h2 class="fw-bold mb-1">Dashboard</h2>
            <p class="text-muted mb-0">እንኳን ደህና መጡ፣ የአማርኛ ኢዲዮም ምደባ ስርዓት</p>
          </div>
          <span class="badge bg-primary rounded-pill px-3 py-2 mt-2 mt-md-0">Signed in</span>
        </div>

        <section id="overview" class="mb-4">
          <h4 class="fw-bold mb-3">Dataset Summary</h4>
          <div class="row g-4">
            <div class="col-md-6 col-xl-3">
              <div class="card shadow-sm border-0 h-100">
                <div class="card-body">
                  <h6 class="text-muted">Total entries</h6>
                  <h3 class="fw-bold">{{ total_rows }}</h3>
                </div>
              </div>
            </div>
            <div class="col-md-6 col-xl-3">
              <div class="card shadow-sm border-0 h-100">
                <div class="card-body">
                  <h6 class="text-muted">Unique idioms</h6>
                  <h3 class="fw-bold">{{ unique_idioms }}</h3>
                </div>
              </div>
            </div>
            <div class="col-md-6 col-xl-3">
              <div class="card shadow-sm border-0 h-100">
                <div class="card-body">
                  <h6 class="text-muted">Positive labels</h6>
                  <h3 class="fw-bold">{{ positive_labels }}</h3>
                </div>
              </div>
            </div>
            <div class="col-md-6 col-xl-3">
              <div class="card shadow-sm border-0 h-100">
                <div class="card-body">
                  <h6 class="text-muted">Negative labels</h6>
                  <h3 class="fw-bold">{{ negative_labels }}</h3>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="dataset" class="mb-4">
          <div class="row g-4">
            <div class="col-lg-7">
              <div class="card shadow-sm border-0 h-100">
                <div class="card-body">
                  <h5 class="card-title">Label distribution</h5>
                  <canvas id="labelChart"></canvas>
                </div>
              </div>
            </div>
            <div class="col-lg-5">
              <div class="card shadow-sm border-0 h-100">
                <div class="card-body">
                  <h5 class="card-title">Recent entries</h5>
                  <ul class="list-group list-group-flush">
                    {% for item in sample_rows %}
                    <li class="list-group-item px-0">
                      <strong>{{ item.text }}</strong><br>
                      <small class="text-muted">Label: {{ item.label }}</small>
                    </li>
                    {% endfor %}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="models" class="mb-4">
          <div class="row g-4">
            <div class="col-lg-7">
              <div class="card shadow-sm border-0 h-100">
                <div class="card-body">
                  <h5 class="card-title">Model performance</h5>
                  <canvas id="modelChart"></canvas>
                </div>
              </div>
            </div>
            <div class="col-lg-5">
              <div class="card shadow-sm border-0 h-100">
                <div class="card-body">
                  <h5 class="card-title">Model summary</h5>
                  <p class="text-muted mb-3">The dashboard is connected to the repository’s dataset and a small model-results file.</p>
                  <ul class="list-unstyled">
                    <li><strong>Accuracy:</strong> {{ model_results.accuracy }}</li>
                    <li><strong>Precision:</strong> {{ model_results.precision }}</li>
                    <li><strong>Recall:</strong> {{ model_results.recall }}</li>
                    <li><strong>F1 Score:</strong> {{ model_results.f1_score }}</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="idioms" class="mb-4">
          <div class="card shadow-sm border-0">
            <div class="card-body">
              <div class="d-flex flex-column flex-md-row justify-content-between align-items-md-center mb-3">
                <h5 class="card-title mb-2 mb-md-0">All Idioms</h5>
                <form method="get" class="d-flex gap-2 flex-wrap">
                  <input type="text" class="form-control" name="search" value="{{ search_term }}" placeholder="Search idiom or label">
                  <select class="form-select" name="label">
                    <option value="">All labels</option>
                    <option value="1" {% if selected_label == '1' %}selected{% endif %}>Positive</option>
                    <option value="0" {% if selected_label == '0' %}selected{% endif %}>Negative</option>
                  </select>
                  <select class="form-select" name="sort">
                    <option value="text" {% if sort_by == 'text' %}selected{% endif %}>Sort by text</option>
                    <option value="label" {% if sort_by == 'label' %}selected{% endif %}>Sort by label</option>
                  </select>
                  <button type="submit" class="btn btn-outline-primary">Filter</button>
                </form>
              </div>
              <div class="table-responsive">
                <table class="table table-hover align-middle">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Idiom</th>
                      <th>Label</th>
                    </tr>
                  </thead>
                  <tbody>
                    {% for item in paginated_rows %}
                    <tr>
                      <td>{{ item.index }}</td>
                      <td>{{ item.text }}</td>
                      <td>{{ item.label }}</td>
                    </tr>
                    {% endfor %}
                  </tbody>
                </table>
              </div>
              <nav aria-label="Page navigation">
                <ul class="pagination justify-content-center mt-3">
                  {% if current_page > 1 %}
                  <li class="page-item">
                    <a class="page-link" href="{{ url_for('dashboard', search=search_term, label=selected_label, sort=sort_by, page=current_page - 1) }}">Previous</a>
                  </li>
                  {% endif %}
                  {% for page_num in page_links %}
                    {% if page_num == '...' %}
                    <li class="page-item disabled">
                      <span class="page-link">&hellip;</span>
                    </li>
                    {% else %}
                    <li class="page-item {% if page_num == current_page %}active{% endif %}">
                      <a class="page-link" href="{{ url_for('dashboard', search=search_term, label=selected_label, sort=sort_by, page=page_num) }}">{{ page_num }}</a>
                    </li>
                    {% endif %}
                  {% endfor %}
                  {% if current_page < total_pages %}
                  <li class="page-item">
                    <a class="page-link" href="{{ url_for('dashboard', search=search_term, label=selected_label, sort=sort_by, page=current_page + 1) }}">Next</a>
                  </li>
                  {% endif %}
                </ul>
              </nav>
            </div>
          </div>
        </section>

        <section id="predict">
          <div class="card shadow-sm border-0">
            <div class="card-body">
              <h5 class="card-title">Predict a New Idiom</h5>
              <form method="post" action="{{ url_for('predict') }}" class="row g-3">
                <div class="col-md-8">
                  <input type="text" class="form-control" name="idiom" placeholder="Enter an Amharic idiom" required>
                </div>
                <div class="col-md-4">
                  <button type="submit" class="btn btn-primary w-100">Predict</button>
                </div>
              </form>
              {% if prediction_result %}
              <div class="alert alert-info mt-3 mb-0">
                <div class="d-flex justify-content-between align-items-start">
                  <div>
                    <p class="mb-1"><strong>Prediction:</strong> {{ prediction_result.label }}</p>
                    <p class="mb-1"><strong>Confidence:</strong> {{ prediction_result.confidence }}%</p>
                  </div>
                  <a href="{{ url_for('download_results', search=search_term, label=selected_label, sort=sort_by) }}" class="btn btn-sm btn-secondary">Download results</a>
                </div>
                {% if prediction_result.class_probs %}
                <div class="mt-3">
                  <h6>Class probabilities</h6>
                  <ul class="list-unstyled mb-0">
                    {% for cls, prob in prediction_result.class_probs.items() %}
                    <li><strong>{{ cls }}:</strong> {{ prob }}%</li>
                    {% endfor %}
                  </ul>
                </div>
                {% endif %}
              </div>
              {% endif %}
            </div>
          </div>
        </section>
      </main>
    </div>
  </div>

  <script>
    const labelChart = new Chart(document.getElementById('labelChart'), {
      type: 'bar',
      data: {
        labels: {{ label_labels|tojson }},
        datasets: [{
          label: 'Label counts',
          data: {{ label_values|tojson }},
          backgroundColor: ['#2563eb', '#60a5fa']
        }]
      },
      options: { responsive: true, plugins: { legend: { display: false } } }
    });

    const modelChart = new Chart(document.getElementById('modelChart'), {
      type: 'radar',
      data: {
        labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        datasets: [{
          label: 'Model metrics',
          data: [{{ model_results.accuracy }}, {{ model_results.precision }}, {{ model_results.recall }}, {{ model_results.f1_score }}],
          backgroundColor: 'rgba(37, 99, 235, 0.2)',
          borderColor: '#2563eb'
        }]
      },
      options: { responsive: true }
    });
  </script>
</body>
</html>
"""


def build_classifier():
    with open(DATASET_PATH, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))

    texts = [row.get('text', '').strip() for row in rows if row.get('text', '').strip()]
    labels = [int(row.get('label', 0)) for row in rows if row.get('text', '').strip()]

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ('clf', LogisticRegression(max_iter=3000))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def load_model():
    if os.path.exists(MODEL_CACHE_PATH):
        return joblib.load(MODEL_CACHE_PATH)

    model = build_classifier()
    joblib.dump(model, MODEL_CACHE_PATH)
    return model


MODEL = load_model()


def predict_label(text):
    if not text or not text.strip():
        return {'label': 'Unknown', 'confidence': 0.0, 'class_probs': {}}

    cleaned = text.strip()
    prediction = int(MODEL.predict([cleaned])[0])
    probabilities = MODEL.predict_proba([cleaned])[0]
    class_probs = {str(cls): round(prob * 100, 2) for cls, prob in zip(MODEL.classes_, probabilities)}
    return {
        'label': 'Positive' if prediction == 1 else 'Negative',
        'confidence': round(max(probabilities) * 100, 2),
        'class_probs': class_probs,
    }


def load_dashboard_context(search_term='', selected_label='', prediction_result=None, page=1, sort_by='text'):
    with open(DATASET_PATH, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))

    filtered_rows = rows
    if search_term:
        search_term = search_term.strip().lower()
        filtered_rows = [
            row for row in filtered_rows
            if search_term in row.get('text', '').lower()
            or search_term in str(row.get('label', '')).lower()
        ]
    if selected_label:
        filtered_rows = [row for row in filtered_rows if row.get('label') == selected_label]

    if sort_by == 'label':
        filtered_rows = sorted(filtered_rows, key=lambda item: item.get('label', ''))
    else:
        filtered_rows = sorted(filtered_rows, key=lambda item: item.get('text', ''))

    label_counts = {}
    for row in rows:
        label = row.get('label', 'unknown')
        label_counts[label] = label_counts.get(label, 0) + 1

    label_labels = list(label_counts.keys())
    label_values = [label_counts[label] for label in label_labels]

    sample_rows = rows[:5]
    positive_labels = label_counts.get('1', 0)
    negative_labels = label_counts.get('0', 0)

    with open(MODEL_RESULTS_PATH, encoding='utf-8') as handle:
        model_results = json.load(handle)

    page_size = 15
    total_pages = max(1, (len(filtered_rows) + page_size - 1) // page_size)
    current_page = max(1, min(int(page), total_pages))
    start = (current_page - 1) * page_size
    end = start + page_size
    paginated_rows = []
    for index, row in enumerate(filtered_rows[start:end], start=start + 1):
        paginated_rows.append({
            'index': index,
            'text': row.get('text', ''),
            'label': row.get('label', ''),
        })

    # show up to 5 page links around current page
    if total_pages <= 7:
        page_links = list(range(1, total_pages + 1))
    else:
        page_links = [1]
        if current_page > 4:
            page_links.append('...')
        for p in range(max(2, current_page - 2), min(total_pages, current_page + 2) + 1):
            page_links.append(p)
        if current_page < total_pages - 3:
            page_links.append('...')
        page_links.append(total_pages)

    return {
        'total_rows': len(rows),
        'unique_idioms': len({row.get('text', '').strip() for row in rows if row.get('text')}),
        'positive_labels': positive_labels,
        'negative_labels': negative_labels,
        'label_labels': label_labels,
        'label_values': label_values,
        'sample_rows': sample_rows,
        'model_results': model_results,
        'idiom_rows': filtered_rows,
        'paginated_rows': paginated_rows,
        'search_term': search_term,
        'selected_label': selected_label,
        'prediction_result': prediction_result,
        'current_page': current_page,
        'total_pages': total_pages,
        'page_links': page_links,
        'sort_by': sort_by,
    }


@app.route('/download', methods=['GET'])
def download_results():
    search_term = request.args.get('search', '')
    selected_label = request.args.get('label', '')
    sort_by = request.args.get('sort', 'text')

    with open(DATASET_PATH, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))

    filtered_rows = rows
    if search_term:
        search_term = search_term.strip().lower()
        filtered_rows = [
            row for row in filtered_rows
            if search_term in row.get('text', '').lower()
            or search_term in str(row.get('label', '')).lower()
        ]
    if selected_label:
        filtered_rows = [row for row in filtered_rows if row.get('label') == selected_label]

    if sort_by == 'label':
        filtered_rows = sorted(filtered_rows, key=lambda item: item.get('label', ''))
    else:
        filtered_rows = sorted(filtered_rows, key=lambda item: item.get('text', ''))

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['text', 'label'])
    for row in filtered_rows:
        writer.writerow([row.get('text', ''), row.get('label', '')])

    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=idiom_results.csv'
    response.headers['Content-Type'] = 'text/csv; charset=utf-8'
    return response


@app.route('/dashboard', methods=['GET'])
def dashboard():
    search_term = request.args.get('search', '')
    selected_label = request.args.get('label', '')
    sort_by = request.args.get('sort', 'text')
    page = request.args.get('page', 1, type=int)
    return render_template_string(DASHBOARD_HTML, **load_dashboard_context(search_term, selected_label, page=page, sort_by=sort_by))


@app.route('/predict', methods=['POST'])
def predict():
    idiom = request.form.get('idiom', '').strip()
    if not idiom:
        prediction_result = {'label': 'Please enter an idiom.', 'confidence': 0.0}
    else:
        prediction_result = predict_label(idiom)
    return render_template_string(DASHBOARD_HTML, **load_dashboard_context(prediction_result=prediction_result))


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        if username == 'admin' and password == 'admin123':
            return dashboard()
        error = 'Invalid username or password'
    return render_template_string(LOGIN_HTML, error=error)


if __name__ == '__main__':
    app.run(debug=True)
