import React, { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "http://localhost:8000";

function App() {
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState("checking");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const [formData, setFormData] = useState({
    age: 43,
    sex: 1,
    cp: 0,
    trestbps: 120,
    chol: 198,
    fbs: 0,
    restecg: 0,
    thalach: 149,
    exang: 0,
    oldpeak: 1.8,
    slope: 1,
    ca: 0,
    thal: 3,
  });

  const featureDescriptions = {
    age: "Age (years)",
    sex: "Sex (0=Female, 1=Male)",
    cp: "Chest Pain Type (0-3)",
    trestbps: "Resting Blood Pressure (mmHg)",
    chol: "Serum Cholesterol (mg/dl)",
    fbs: "Fasting Blood Sugar > 120 (0=No, 1=Yes)",
    restecg: "Resting ECG (0-2)",
    thalach: "Max Heart Rate",
    exang: "Exercise Angina (0=No, 1=Yes)",
    oldpeak: "ST Depression",
    slope: "ST Slope (0-2)",
    ca: "Healthy Vessels (0-3)",
    thal: "Thalassemia (0-3)",
  };

  // Check API status on mount
  useEffect(() => {
    checkAPIStatus();
  }, []);

  const checkAPIStatus = async () => {
    try {
      await axios.get(`${API_URL}/health`);
      setApiStatus("healthy");
    } catch (err) {
      setApiStatus("unhealthy");
      setError(
        "❌ Cannot connect to API. Make sure FastAPI server is running on http://localhost:8000",
      );
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: name === "oldpeak" ? parseFloat(value) : parseInt(value),
    }));
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData);
      setResult(response.data);
    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message;
      setError(`⚠️ Prediction failed: ${errorMsg}`);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData({
      age: 43,
      sex: 1,
      cp: 0,
      trestbps: 120,
      chol: 198,
      fbs: 0,
      restecg: 0,
      thalach: 149,
      exang: 0,
      oldpeak: 1.8,
      slope: 1,
      ca: 0,
      thal: 3,
    });
    setResult(null);
    setError(null);
  };

  return (
    <div className="container">
      <header className="header">
        <h1>🏥 Heart Disease Prediction</h1>
        <p className="subtitle">AI-Powered Risk Assessment</p>

        <div className={`status-badge ${apiStatus}`}>
          {apiStatus === "healthy" ? "✓ API Connected" : "✗ API Disconnected"}
        </div>
      </header>

      {error && <div className="error-box">{error}</div>}

      <div className="main-content">
        <form onSubmit={handleSubmit} className="prediction-form">
          <section className="form-section">
            <h2>Patient Information</h2>

            <div className="form-grid">
              {/* Row 1: Age and Sex */}
              <div className="form-group">
                <label htmlFor="age">{featureDescriptions.age}</label>
                <input
                  type="number"
                  id="age"
                  name="age"
                  value={formData.age}
                  onChange={handleInputChange}
                  min="18"
                  max="120"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="sex">{featureDescriptions.sex}</label>
                <select
                  id="sex"
                  name="sex"
                  value={formData.sex}
                  onChange={handleInputChange}
                  required
                >
                  <option value="0">Female</option>
                  <option value="1">Male</option>
                </select>
              </div>

              {/* Row 2: CP and Blood Pressure */}
              <div className="form-group">
                <label htmlFor="cp">{featureDescriptions.cp}</label>
                <select
                  id="cp"
                  name="cp"
                  value={formData.cp}
                  onChange={handleInputChange}
                  required
                >
                  <option value="0">Typical Angina</option>
                  <option value="1">Atypical Angina</option>
                  <option value="2">Non-Anginal Pain</option>
                  <option value="3">Asymptomatic</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="trestbps">{featureDescriptions.trestbps}</label>
                <input
                  type="number"
                  id="trestbps"
                  name="trestbps"
                  value={formData.trestbps}
                  onChange={handleInputChange}
                  min="80"
                  max="220"
                  required
                />
              </div>

              {/* Row 3: Cholesterol and FBS */}
              <div className="form-group">
                <label htmlFor="chol">{featureDescriptions.chol}</label>
                <input
                  type="number"
                  id="chol"
                  name="chol"
                  value={formData.chol}
                  onChange={handleInputChange}
                  min="100"
                  max="600"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="fbs">{featureDescriptions.fbs}</label>
                <select
                  id="fbs"
                  name="fbs"
                  value={formData.fbs}
                  onChange={handleInputChange}
                  required
                >
                  <option value="0">No</option>
                  <option value="1">Yes</option>
                </select>
              </div>

              {/* Row 4: ECG and Heart Rate */}
              <div className="form-group">
                <label htmlFor="restecg">{featureDescriptions.restecg}</label>
                <select
                  id="restecg"
                  name="restecg"
                  value={formData.restecg}
                  onChange={handleInputChange}
                  required
                >
                  <option value="0">Normal</option>
                  <option value="1">ST-T Abnormality</option>
                  <option value="2">LV Hypertrophy</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="thalach">{featureDescriptions.thalach}</label>
                <input
                  type="number"
                  id="thalach"
                  name="thalach"
                  value={formData.thalach}
                  onChange={handleInputChange}
                  min="50"
                  max="210"
                  required
                />
              </div>

              {/* Row 5: Exercise Angina and ST Depression */}
              <div className="form-group">
                <label htmlFor="exang">{featureDescriptions.exang}</label>
                <select
                  id="exang"
                  name="exang"
                  value={formData.exang}
                  onChange={handleInputChange}
                  required
                >
                  <option value="0">No</option>
                  <option value="1">Yes</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="oldpeak">{featureDescriptions.oldpeak}</label>
                <input
                  type="number"
                  id="oldpeak"
                  name="oldpeak"
                  value={formData.oldpeak}
                  onChange={handleInputChange}
                  min="0"
                  max="10"
                  step="0.1"
                  required
                />
              </div>

              {/* Row 6: ST Slope and Vessels */}
              <div className="form-group">
                <label htmlFor="slope">{featureDescriptions.slope}</label>
                <select
                  id="slope"
                  name="slope"
                  value={formData.slope}
                  onChange={handleInputChange}
                  required
                >
                  <option value="0">Upsloping</option>
                  <option value="1">Flat</option>
                  <option value="2">Downsloping</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="ca">{featureDescriptions.ca}</label>
                <input
                  type="number"
                  id="ca"
                  name="ca"
                  value={formData.ca}
                  onChange={handleInputChange}
                  min="0"
                  max="3"
                  required
                />
              </div>

              {/* Row 7: Thalassemia */}
              <div className="form-group full-width">
                <label htmlFor="thal">{featureDescriptions.thal}</label>
                <select
                  id="thal"
                  name="thal"
                  value={formData.thal}
                  onChange={handleInputChange}
                  required
                >
                  <option value="0">Normal</option>
                  <option value="1">Fixed Defect</option>
                  <option value="2">Reversible Defect</option>
                  <option value="3">Detected</option>
                </select>
              </div>
            </div>
          </section>

          <div className="button-group">
            <button
              type="submit"
              disabled={loading || apiStatus !== "healthy"}
              className="btn btn-primary"
            >
              {loading ? "⏳ Analyzing..." : "🔍 Predict"}
            </button>
            <button
              type="button"
              onClick={handleReset}
              className="btn btn-secondary"
            >
              🔄 Reset
            </button>
          </div>
        </form>

        {/* Results Section */}
        {result && (
          <div
            className={`result-box ${result.disease_present ? "positive" : "negative"}`}
          >
            <h2>
              {result.disease_present
                ? "⚠️ Disease Present"
                : "✅ No Disease Detected"}
            </h2>

            <div className="result-grid">
              <div className="result-item">
                <span className="label">Prediction</span>
                <span className="value">{result.prediction_label}</span>
              </div>

              <div className="result-item">
                <span className="label">Confidence</span>
                <span className="value">
                  {(result.confidence * 100).toFixed(2)}%
                </span>
              </div>

              <div className="result-item">
                <span className="label">Risk Level</span>
                <span
                  className={`value risk-${result.risk_level.toLowerCase().replace(" ", "-")}`}
                >
                  {result.risk_level}
                </span>
              </div>
            </div>

            <div className="probability-chart">
              <h3>Probability Distribution</h3>
              <div className="chart">
                <div className="bar-group">
                  <div className="bar-item">
                    <div className="bar-label">No Disease</div>
                    <div className="bar-container">
                      <div
                        className="bar bar-green"
                        style={{
                          width: `${result.probability_no_disease * 100}%`,
                        }}
                      >
                        {(result.probability_no_disease * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  <div className="bar-item">
                    <div className="bar-label">Disease Present</div>
                    <div className="bar-container">
                      <div
                        className="bar bar-red"
                        style={{
                          width: `${result.probability_disease * 100}%`,
                        }}
                      >
                        {(result.probability_disease * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="result-message">
              {result.disease_present ? (
                <p>
                  ⚠️ <strong>This patient may have heart disease.</strong>{" "}
                  Please consult a healthcare professional for proper diagnosis
                  and treatment.
                </p>
              ) : (
                <p>
                  ✅ <strong>No signs of heart disease detected.</strong>{" "}
                  However, regular health checkups are recommended.
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      <footer className="footer">
        <p>
          🏥 Heart Disease Prediction System | Powered by KNN Machine Learning
          Model
        </p>
        <p>
          ⚠️ Disclaimer: This is a predictive tool for research purposes only.
          Always consult a licensed healthcare professional.
        </p>
      </footer>
    </div>
  );
}

export default App;
