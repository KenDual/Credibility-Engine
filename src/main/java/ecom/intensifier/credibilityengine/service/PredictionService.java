package ecom.intensifier.credibilityengine.service;

import ecom.intensifier.credibilityengine.DTO.*;
import ecom.intensifier.credibilityengine.entity.Prediction;
import ecom.intensifier.credibilityengine.repository.PredictionRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

/**
 * Service layer for fraud prediction business logic
 * Handles prediction workflow and data persistence
 */
@Service
@Transactional
public class PredictionService {

    private static final Logger logger = LoggerFactory.getLogger(PredictionService.class);

    private final PythonApiService pythonApiService;
    private final PredictionRepository predictionRepository;

    @Autowired
    public PredictionService(PythonApiService pythonApiService,
            PredictionRepository predictionRepository) {
        this.pythonApiService = pythonApiService;
        this.predictionRepository = predictionRepository;
    }

    /**
     * Main method to perform fraud prediction
     * 1. Call Python API
     * 2. Save result to database
     * 3. Return response
     * 
     * @param form Fraud check form from user
     * @return Prediction response
     */
    public PredictionResponse predictFraud(FraudCheckForm form) {
        logger.info("Starting fraud prediction for order: {}", form.getOrderId());

        try {
            // Convert form to API request
            PredictionRequest apiRequest = form.toApiRequest();

            // Call Python API
            PredictionResponse apiResponse = pythonApiService.getPrediction(apiRequest);

            if (apiResponse.isSuccess()) {
                // Save to database
                Prediction savedPrediction = savePrediction(form, apiResponse);
                logger.info("Prediction saved with ID: {} for order: {}",
                        savedPrediction.getId(), savedPrediction.getOrderId());
            }

            return apiResponse;

        } catch (Exception e) {
            logger.error("Error during fraud prediction: {}", e.getMessage(), e);

            // Return error response
            PredictionResponse errorResponse = new PredictionResponse();
            errorResponse.setSuccess(false);
            errorResponse.setError("Prediction failed: " + e.getMessage());
            return errorResponse;
        }
    }

    /**
     * Save prediction result to database
     */
    private Prediction savePrediction(FraudCheckForm form, PredictionResponse response) {
        Prediction prediction = new Prediction();

        // Order information
        prediction.setOrderId(response.getOrderId());
        prediction.setProductId(form.getProductId());
        prediction.setUserId(form.getUserId());

        // Prediction results
        prediction.setPredictionResult(response.getPrediction());
        prediction.setPredictionBinary(response.getPredictionBinary());
        prediction.setProbability(BigDecimal.valueOf(response.getProbability()));
        prediction.setRiskLevel(response.getRiskLevel());
        prediction.setRiskScore(response.getRiskScore());
        prediction.setConfidence(response.getConfidence());

        // Order details
        prediction.setProductCategory(form.getProductCategory());
        prediction.setProductPrice(form.getProductPrice());
        prediction.setOrderQuantity(form.getOrderQuantity());
        prediction.setDiscountApplied(form.getDiscountApplied());

        // Calculate derived fields
        BigDecimal totalValue = form.getProductPrice()
                .multiply(BigDecimal.valueOf(form.getOrderQuantity()));
        prediction.setTotalOrderValue(totalValue);

        if (totalValue.compareTo(BigDecimal.ZERO) > 0) {
            BigDecimal discountPct = form.getDiscountApplied()
                    .divide(totalValue, 4, BigDecimal.ROUND_HALF_UP)
                    .multiply(BigDecimal.valueOf(100));
            prediction.setDiscountPercentage(discountPct);
        }

        // User information
        prediction.setUserAge(form.getUserAge());
        prediction.setUserGender(form.getUserGender());
        prediction.setUserLocation(form.getUserLocation());

        // Transaction details
        prediction.setPaymentMethod(form.getPaymentMethod());
        prediction.setShippingMethod(form.getShippingMethod());
        prediction.setOrderDate(form.getOrderDate());

        // Metadata
        prediction.setModelVersion(response.getModelVersion());
        prediction.setPredictedAt(LocalDateTime.now());

        return predictionRepository.save(prediction);
    }

    /**
     * Get all predictions with pagination
     */
    public Page<Prediction> getAllPredictions(int page, int size) {
        Pageable pageable = PageRequest.of(page, size, Sort.by("predictedAt").descending());
        return predictionRepository.findAll(pageable);
    }

    /**
     * Get recent predictions (last 100)
     */
    public Page<Prediction> getRecentPredictions(int page, int size) {
        Pageable pageable = PageRequest.of(page, size, Sort.by("predictedAt").descending());
        return predictionRepository.findAll(pageable);
    }

    /**
     * Get prediction by ID
     */
    public Optional<Prediction> getPredictionById(Long id) {
        return predictionRepository.findById(id);
    }

    /**
     * Get prediction by order ID
     */
    public Optional<Prediction> getPredictionByOrderId(String orderId) {
        return predictionRepository.findByOrderId(orderId);
    }

    /**
     * Get predictions by user ID
     */
    public List<Prediction> getPredictionsByUserId(String userId) {
        return predictionRepository.findByUserId(userId);
    }

    /**
     * Get predictions by risk level
     */
    public Page<Prediction> getPredictionsByRiskLevel(String riskLevel, int page, int size) {
        Pageable pageable = PageRequest.of(page, size, Sort.by("predictedAt").descending());
        return predictionRepository.findByRiskLevel(riskLevel, pageable);
    }

    /**
     * Search predictions by order ID or user ID
     */
    public Page<Prediction> searchPredictions(String searchTerm, int page, int size) {
        Pageable pageable = PageRequest.of(page, size, Sort.by("predictedAt").descending());
        return predictionRepository.searchByOrderOrUserId(searchTerm, pageable);
    }

    /**
     * Get statistics for dashboard
     */
    public StatisticsResponse getStatistics() {
        Long total = predictionRepository.getTotalCount();
        Long lowRisk = predictionRepository.countByRiskLevel("LOW");
        Long mediumRisk = predictionRepository.countByRiskLevel("MEDIUM");
        Long highRisk = predictionRepository.countByRiskLevel("HIGH");
        Long trusted = predictionRepository.countByPredictionResult("Trusted");
        Long uncertain = predictionRepository.countByPredictionResult("Uncertain");
        Long untrustworthy = predictionRepository.countByPredictionResult("Untrustworthy");
        Double avgRiskScore = predictionRepository.getAverageRiskScore();

        StatisticsResponse stats = new StatisticsResponse();
        stats.setTotalPredictions(total != null ? total : 0L);
        stats.setLowRiskCount(lowRisk != null ? lowRisk : 0L);
        stats.setMediumRiskCount(mediumRisk != null ? mediumRisk : 0L);
        stats.setHighRiskCount(highRisk != null ? highRisk : 0L);
        stats.setTrustedCount(trusted != null ? trusted : 0L);
        stats.setUncertainCount(uncertain != null ? uncertain : 0L);
        stats.setUntrustworthyCount(untrustworthy != null ? untrustworthy : 0L);
        stats.setAverageRiskScore(avgRiskScore != null ? avgRiskScore : 0.0);

        // Calculate percentages
        if (total != null && total > 0) {
            stats.setLowRiskPercentage((lowRisk != null ? lowRisk : 0) * 100.0 / total);
            stats.setMediumRiskPercentage((mediumRisk != null ? mediumRisk : 0) * 100.0 / total);
            stats.setHighRiskPercentage((highRisk != null ? highRisk : 0) * 100.0 / total);
        }

        return stats;
    }

    /**
     * Get high risk predictions
     */
    public List<Prediction> getHighRiskPredictions() {
        return predictionRepository.findHighRiskPredictions();
    }

    /**
     * Get recent high risk predictions with pagination
     */
    public Page<Prediction> getRecentHighRiskPredictions(int page, int size) {
        Pageable pageable = PageRequest.of(page, size);
        return predictionRepository.findRecentHighRiskPredictions(pageable);
    }

    /**
     * Delete prediction by ID
     */
    public void deletePrediction(Long id) {
        predictionRepository.deleteById(id);
        logger.info("Deleted prediction with ID: {}", id);
    }

    /**
     * Check if Python API is healthy
     */
    public boolean isPythonApiHealthy() {
        return pythonApiService.isApiHealthy();
    }

    /**
     * Get model information
     */
    public java.util.Map<String, Object> getModelInfo() {
        return pythonApiService.getModelInfo();
    }

    /**
     * Get total prediction count
     */
    public Long getTotalCount() {
        return predictionRepository.getTotalCount();
    }

    /**
     * Get predictions within date range
     */
    public List<Prediction> getPredictionsBetweenDates(LocalDateTime startDate, LocalDateTime endDate) {
        return predictionRepository.findPredictionsBetweenDates(startDate, endDate);
    }
}