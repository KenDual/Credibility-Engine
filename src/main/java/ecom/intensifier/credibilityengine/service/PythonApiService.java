package ecom.intensifier.credibilityengine.service;

import ecom.intensifier.credibilityengine.DTO.PredictionRequest;
import ecom.intensifier.credibilityengine.DTO.PredictionResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

/**
 * Service for calling Python Flask API
 * Handles all communication with ML model API
 */
@Service
public class PythonApiService {

    private static final Logger logger = LoggerFactory.getLogger(PythonApiService.class);

    private final RestTemplate restTemplate;

    @Value("${python.api.base-url:http://localhost:5000}")
    private String pythonApiBaseUrl;

    @Value("${python.api.predict-endpoint:/api/predict}")
    private String predictEndpoint;

    @Value("${python.api.health-endpoint:/api/health}")
    private String healthEndpoint;

    @Value("${python.api.model-info-endpoint:/api/model/info}")
    private String modelInfoEndpoint;

    public PythonApiService() {
        this.restTemplate = new RestTemplate();
    }

    /**
     * Call Python API to get fraud prediction
     * 
     * @param request Prediction request with order details
     * @return Prediction response from ML model
     * @throws RestClientException if API call fails
     */
    public PredictionResponse getPrediction(PredictionRequest request) {
        String url = pythonApiBaseUrl + predictEndpoint;

        logger.info("Calling Python API: {} with order: {}", url, request.getOrderId());

        try {
            // Create headers
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            // Create request entity
            HttpEntity<PredictionRequest> entity = new HttpEntity<>(request, headers);

            // Call API
            ResponseEntity<PredictionResponse> response = restTemplate.exchange(
                    url,
                    HttpMethod.POST,
                    entity,
                    PredictionResponse.class);

            PredictionResponse predictionResponse = response.getBody();

            if (predictionResponse != null && predictionResponse.isSuccess()) {
                logger.info("Prediction successful for order: {} - Risk: {} ({}%)",
                        request.getOrderId(),
                        predictionResponse.getRiskLevel(),
                        predictionResponse.getRiskScore());
                return predictionResponse;
            } else {
                logger.error("Prediction failed for order: {} - Error: {}",
                        request.getOrderId(),
                        predictionResponse != null ? predictionResponse.getError() : "Unknown error");
                throw new RuntimeException("Prediction API returned unsuccessful response");
            }

        } catch (RestClientException e) {
            logger.error("Error calling Python API for order: {} - Error: {}",
                    request.getOrderId(), e.getMessage(), e);
            throw new RuntimeException("Failed to connect to ML prediction service: " + e.getMessage(), e);
        }
    }

    /**
     * Check if Python API is healthy and running
     * 
     * @return true if API is healthy, false otherwise
     */
    public boolean isApiHealthy() {
        String url = pythonApiBaseUrl + healthEndpoint;

        try {
            ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                Map<String, Object> body = response.getBody();
                String status = (String) body.get("status");
                Boolean modelLoaded = (Boolean) body.get("model_loaded");

                boolean healthy = "healthy".equals(status) && Boolean.TRUE.equals(modelLoaded);
                logger.info("Python API health check: {} (model loaded: {})", status, modelLoaded);
                return healthy;
            }

            return false;

        } catch (Exception e) {
            logger.error("Python API health check failed: {}", e.getMessage());
            return false;
        }
    }

    /**
     * Get model information from Python API
     * 
     * @return Model metadata as Map
     */
    public Map<String, Object> getModelInfo() {
        String url = pythonApiBaseUrl + modelInfoEndpoint;

        try {
            ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);

            if (response.getStatusCode() == HttpStatus.OK) {
                logger.info("Retrieved model info from Python API");
                return response.getBody();
            }

            return new HashMap<>();

        } catch (Exception e) {
            logger.error("Error getting model info: {}", e.getMessage());
            return new HashMap<>();
        }
    }

    /**
     * Get Python API base URL
     */
    public String getApiUrl() {
        return pythonApiBaseUrl;
    }

    /**
     * Validate Python API connection on startup
     * 
     * @return true if connection successful
     */
    public boolean validateConnection() {
        logger.info("Validating connection to Python API at: {}", pythonApiBaseUrl);

        boolean healthy = isApiHealthy();

        if (healthy) {
            Map<String, Object> modelInfo = getModelInfo();
            logger.info("Python API connection validated successfully. Model: {}",
                    modelInfo.get("model_name"));
        } else {
            logger.warn("Python API connection validation failed. Please ensure Flask API is running at: {}",
                    pythonApiBaseUrl);
        }

        return healthy;
    }
}