import React, { useState, useEffect } from 'react';
import { Upload, Image, Activity, BarChart3, History, Leaf, X, RefreshCw } from 'lucide-react';
import "./App.css";

const API_URL = 'http://localhost:5000/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('classify');
  const [selectedModel, setSelectedModel] = useState('all');
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [segmentations, setSegmentations] = useState([]);
  const [statistics, setStatistics] = useState(null);

  useEffect(() => {
    fetchModels();
    fetchStatistics();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await fetch(`${API_URL}/models`);
      const data = await response.json();
      setAvailableModels(data);
    } catch (error) {
      console.error('Error al cargar modelos:', error);
    }
  };

  const fetchStatistics = async () => {
    try {
      const response = await fetch(`${API_URL}/statistics`);
      const data = await response.json();
      setStatistics(data);
    } catch (error) {
      console.error('Error al cargar estadísticas:', error);
    }
  };

  const fetchPredictions = async () => {
    try {
      const response = await fetch(`${API_URL}/predictions?limit=10`);
      const data = await response.json();
      setPredictions(data);
    } catch (error) {
      console.error('Error al cargar predicciones:', error);
    }
  };

  const fetchSegmentations = async () => {
    try {
      const response = await fetch(`${API_URL}/segmentations?limit=10`);
      const data = await response.json();
      setSegmentations(data);
    } catch (error) {
      console.error('Error al cargar segmentaciones:', error);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults(null);
    }
  };

  const handleRemoveImage = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResults(null);
    const fileInput = document.getElementById('file-upload');
    if (fileInput) fileInput.value = '';
  };

  const handleClassify = async () => {
    if (!selectedFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('model', selectedModel);

    try {
      const response = await fetch(`${API_URL}/classify`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResults(data);
      fetchStatistics();
    } catch (error) {
      console.error('Error en clasificación:', error);
      alert('Error al clasificar la imagen');
    } finally {
      setLoading(false);
    }
  };

  const handleSegment = async () => {
    if (!selectedFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch(`${API_URL}/segment`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResults(data);
      fetchStatistics();
    } catch (error) {
      console.error('Error en segmentación:', error);
      alert('Error al segmentar la imagen');
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setResults(null);
    
    if (tab === 'history') {
      fetchPredictions();
    } else if (tab === 'segmentations') {
      fetchSegmentations();
    }
  };

  const getConfidenceClass = (confidence) => {
    if (confidence >= 0.8) return 'high-confidence';
    if (confidence >= 0.5) return 'medium-confidence';
    return 'low-confidence';
  };

  const getClassColor = (prediction) => {
    const colors = {
      'ripe': '#22c55e',
      'unripe': '#eab308',
      'damaged': '#ef4444',
      'old': '#f97316'
    };
    return colors[prediction] || '#6b7280';
  };

  const getClassLabel = (prediction) => {
    const labels = {
      'ripe': 'Maduro',
      'unripe': 'Verde',
      'damaged': 'Dañado',
      'old': 'Viejo'
    };
    return labels[prediction] || prediction;
  };

  const getModelLabel = (model) => {
    const labels = {
      'efficientnetb3': 'EfficientNet-B3',
      'resnet50': 'ResNet-50',
      'inceptionv3': 'Inception-V3',
      'all': 'Todos los modelos'
    };
    return labels[model] || model;
  };



  return (
    <div className="app-container">
      <div className="container">
        <header className="app-header">
          <div className="logo-header">
            <Leaf className="stat-icon green" />
            <h1>Clasificador de Tomates</h1>
            <Leaf className="stat-icon green" />
          </div>
          <p className="subtitle">
            Sistema de clasificación y segmentación con Deep Learning para identificar la condición del tomate
          </p>
        </header>

        {statistics && (
          <div className="stats-grid-two-cards">
            <div className="stat-card">
              <div className="stat-card-content">
                <BarChart3 className="stat-icon blue" />
                <div className="stat-info">
                  <p className="stat-label">Clasificaciones Totales</p>
                  <p className="stat-value">{statistics.total_predictions}</p>
                </div>
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-card-content">
                <Activity className="stat-icon purple" />
                <div className="stat-info">
                  <p className="stat-label">Segmentaciones Totales</p>
                  <p className="stat-value">{statistics.total_segmentations}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="tabs">
          <button
            onClick={() => handleTabChange('classify')}
            className={`tab-button ${activeTab === 'classify' ? 'active' : ''}`}
          >
            <Image size={20} />
            Clasificar
          </button>
          <button
            onClick={() => handleTabChange('segment')}
            className={`tab-button ${activeTab === 'segment' ? 'active purple' : ''}`}
          >
            <Activity size={20} />
            Segmentar
          </button>
          <button
            onClick={() => handleTabChange('history')}
            className={`tab-button ${activeTab === 'history' ? 'active blue' : ''}`}
          >
            <History size={20} />
            Historial
          </button>
        </div>

        <div className="content-card">
          {(activeTab === 'classify' || activeTab === 'segment') && (
            <>
              <div className="upload-section">
                <label className="section-label">
                  Subir Imagen de Tomate
                </label>
                
                {!previewUrl ? (
                  <div className="upload-zone">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileSelect}
                      className="file-input"
                      id="file-upload"
                    />
                    <label htmlFor="file-upload" className="upload-label">
                      <Upload className="upload-icon" />
                      <p className="upload-text">
                        Haz clic para seleccionar una imagen
                      </p>
                      <p className="upload-hint">
                        PNG, JPG hasta 10MB
                      </p>
                    </label>
                  </div>
                ) : (
                  <div className="preview-container">
                    <div className="preview-header">
                      <span className="preview-title">Vista previa</span>
                      <div className="preview-actions">
                        <button 
                          className="action-btn remove-btn"
                          onClick={handleRemoveImage}
                        >
                          <X size={16} />
                          Eliminar
                        </button>
                      </div>
                    </div>
                    <img
                      src={previewUrl}
                      alt="Preview"
                      className="preview-image"
                    />
                  </div>
                )}
              </div>

              {activeTab === 'classify' && (
                <div className="form-group">
                  <label className="section-label">
                    Seleccionar Modelo
                  </label>
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="select"
                  >
                    <option value="all">Todos los modelos</option>
                    {availableModels.map((model) => (
                      <option
                        key={model.name}
                        value={model.name}
                        disabled={!model.available}
                      >
                        {model.display_name} {!model.available && '(No disponible)'}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              <button
                onClick={activeTab === 'classify' ? handleClassify : handleSegment}
                disabled={!selectedFile || loading}
                className={`btn ${activeTab === 'classify' ? 'btn-primary' : 'btn-purple'}`}
              >
                {loading ? (
                  <>
                    <span className="loading-spinner"></span>
                    Procesando...
                  </>
                ) : (
                  activeTab === 'classify' ? 'Clasificar Imagen' : 'Segmentar Imagen'
                )}
              </button>

              {results && activeTab === 'classify' && (
                <div className="results-container">
                  <h3 className="results-title">
                    Resultados de Clasificación
                  </h3>
                  
                  {Object.entries(results.results).map(([modelName, result]) => (
                    <div key={modelName} className="result-card">
                      <h4 className="result-model-name">
                        {getModelLabel(modelName)}
                      </h4>
                      
                      <div className="result-grid">
                        <div className="result-item">
                          <p className="result-item-label">Predicción</p>
                          <p 
                            className="result-item-value"
                            style={{ 
                              color: getClassColor(result.prediction),
                              fontWeight: 'bold'
                            }}
                          >
                            {getClassLabel(result.prediction)}
                          </p>
                        </div>
                        <div className="result-item">
                          <p className="result-item-label">Confianza</p>
                          <p className={`result-item-value ${getConfidenceClass(result.confidence)}`}>
                            {(result.confidence * 100).toFixed(2)}%
                          </p>
                        </div>
                      </div>

                      <div className="probabilities-section">
                        <p className="probabilities-title">Probabilidades por clase:</p>
                        <div>
                          {Object.entries(result.all_probabilities)
                            .sort((a, b) => b[1] - a[1])
                            .map(([className, prob]) => (
                              <div key={className} className="probability-item">
                                <div className="probability-header">
                                  <span className="probability-class">
                                    {getClassLabel(className)}
                                  </span>
                                  <span className="probability-value">
                                    {(prob * 100).toFixed(1)}%
                                  </span>
                                </div>
                                <div className="probability-bar-container">
                                  <div
                                    className="probability-bar"
                                    style={{ 
                                      width: `${prob * 100}%`,
                                      backgroundColor: getClassColor(className)
                                    }}
                                  />
                                </div>
                              </div>
                            ))}
                        </div>
                      </div>

                      <p className="inference-time">
                        Tiempo de inferencia: {result.inference_time}s
                      </p>
                    </div>
                  ))}
                </div>
              )}

              {results && activeTab === 'segment' && (
                <div className="segmentation-results">
                  <h3 className="segmentation-title">
                    Resultado de Segmentación
                  </h3>
                  
                  <div className="image-comparison">
                    <div className="comparison-item">
                      <h4>Imagen Original</h4>
                      <img
                        src={results.original_image}
                        alt="Original"
                      />
                    </div>
                    <div className="comparison-item">
                      <h4>Segmentación</h4>
                      <img
                        src={results.segmented_image}
                        alt="Segmented"
                      />
                    </div>
                  </div>

                  <div className="segmentation-stats">
                    <div className="stats-grid-two">
                      <div className="result-item">
                        <p className="result-item-label">Objetos Detectados</p>
                        <p className="result-item-value">
                          {results.num_objects}
                        </p>
                      </div>
                      <div className="result-item">
                        <p className="result-item-label">Tiempo de Inferencia</p>
                        <p className="result-item-value">
                          {results.inference_time}s
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}

          {activeTab === 'history' && (
            <div className="history-container">
              <h3 className="history-title">
                Historial de Predicciones
              </h3>
              
              {predictions.length === 0 ? (
                <p className="history-empty">
                  No hay predicciones aún
                </p>
              ) : (
                <div className="history-list">
                  {predictions.map((pred) => {
                    return (
                      <div key={pred._id} className="history-item-enhanced">
                        <div className="history-header-section">
                          <div className="history-info">
                            <p className="history-filename">{pred.image_name}</p>
                            <p className="history-timestamp">
                              {new Date(pred.timestamp).toLocaleString('es-ES', {
                                day: '2-digit',
                                month: '2-digit',
                                year: 'numeric',
                                hour: '2-digit',
                                minute: '2-digit'
                              })}
                            </p>
                          </div>

                          <div className="history-meta">
                            <p className="meta-label">Modelo(s) usado(s)</p>
                            <p className="meta-value">
                              {getModelLabel(pred.selected_model)}
                            </p>
                          </div>
                        </div>

                        {/* Resultados detallados de todos los modelos */}
                        <div className="history-results-detail">
                          <p className="detail-title">Resultados de clasificación:</p>
                          {Object.entries(pred.results || {}).map(([modelName, result]) => (
                            <div key={modelName} className="model-result-compact">
                              <div className="model-result-header">
                                <span className="model-name-compact">
                                  {getModelLabel(modelName)}
                                </span>
                                <span 
                                  className="prediction-compact"
                                  style={{ color: getClassColor(result.prediction) }}
                                >
                                  {getClassLabel(result.prediction)}
                                </span>
                                <span className={`confidence-compact ${getConfidenceClass(result.confidence)}`}>
                                  {(result.confidence * 100).toFixed(1)}%
                                </span>
                              </div>
                              
                              {/* Mini barra de probabilidades */}
                              <div className="mini-probabilities">
                                {Object.entries(result.all_probabilities || {})
                                  .sort((a, b) => b[1] - a[1])
                                  .slice(0, 3)
                                  .map(([className, prob]) => (
                                    <div key={className} className="mini-prob-item">
                                      <span className="mini-prob-label">
                                        {getClassLabel(className)}:
                                      </span>
                                      <div className="mini-prob-bar-container">
                                        <div
                                          className="mini-prob-bar"
                                          style={{ 
                                            width: `${prob * 100}%`,
                                            backgroundColor: getClassColor(className)
                                          }}
                                        />
                                      </div>
                                      <span className="mini-prob-value">
                                        {(prob * 100).toFixed(0)}%
                                      </span>
                                    </div>
                                  ))}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}