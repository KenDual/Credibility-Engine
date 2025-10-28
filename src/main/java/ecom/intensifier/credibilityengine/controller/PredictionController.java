package ecom.intensifier.credibilityengine.controller;

import ecom.intensifier.credibilityengine.DTO.*;
import ecom.intensifier.credibilityengine.entity.Prediction;
import ecom.intensifier.credibilityengine.service.PredictionService;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import java.time.LocalDate;
import java.util.HashMap;
import java.util.Map;

@Controller
public class PredictionController {

    private static final Logger logger = LoggerFactory.getLogger(PredictionController.class);

    private final PredictionService predictionService;

    @Autowired
    public PredictionController(PredictionService predictionService) {
        this.predictionService = predictionService;
    }

    // Home page
    @GetMapping("/")
    public String home(Model model) {
        model.addAttribute("apiHealthy", predictionService.isPythonApiHealthy());
        model.addAttribute("totalPredictions", predictionService.getTotalCount());
        return "index";
    }

    // Prediction form page
    @GetMapping("/predict")
    public String showPredictionForm(Model model) {
        model.addAttribute("fraudCheckForm", new FraudCheckForm());
        model.addAttribute("apiHealthy", predictionService.isPythonApiHealthy());

        // Add default date
        FraudCheckForm form = (FraudCheckForm) model.getAttribute("fraudCheckForm");
        if (form != null && form.getOrderDate() == null) {
            form.setOrderDate(LocalDate.now());
        }

        return "predict-form";
    }

    // Predict form submission
    @PostMapping("/predict")
    public String submitPrediction(
            @Valid @ModelAttribute("fraudCheckForm") FraudCheckForm form,
            BindingResult bindingResult,
            Model model,
            RedirectAttributes redirectAttributes) {

        // Validation errors
        if (bindingResult.hasErrors()) {
            model.addAttribute("apiHealthy", predictionService.isPythonApiHealthy());
            return "predict-form";
        }

        try {
            // Check API health
            if (!predictionService.isPythonApiHealthy()) {
                model.addAttribute("error", "Prediction service is currently unavailable. Please try again later.");
                model.addAttribute("apiHealthy", false);
                return "predict-form";
            }

            // Get prediction
            PredictionResponse response = predictionService.predictFraud(form);

            if (response.isSuccess()) {
                // Success - redirect to result page
                redirectAttributes.addFlashAttribute("result", response);
                redirectAttributes.addFlashAttribute("formData", form);
                return "redirect:/result";
            } else {
                // Error from API
                model.addAttribute("error", "Prediction failed: " + response.getError());
                model.addAttribute("apiHealthy", predictionService.isPythonApiHealthy());
                return "predict-form";
            }

        } catch (Exception e) {
            logger.error("Error processing prediction: {}", e.getMessage(), e);
            model.addAttribute("error", "An unexpected error occurred. Please try again.");
            model.addAttribute("apiHealthy", predictionService.isPythonApiHealthy());
            return "predict-form";
        }
    }

    // Result page
    @GetMapping("/result")
    public String showResult(Model model) {
        if (!model.containsAttribute("result")) {
            return "redirect:/predict";
        }

        if (!model.containsAttribute("formData")) {
            model.addAttribute("formData", new FraudCheckForm());
        }
        return "result";
    }

    // History page
    @GetMapping("/history")
    public String showHistory(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size,
            @RequestParam(required = false) String riskLevel,
            @RequestParam(required = false) String search,
            Model model) {

        Page<Prediction> predictions;

        if (search != null && !search.trim().isEmpty()) {
            // Search by order ID or user ID
            predictions = predictionService.searchPredictions(search, page, size);
            model.addAttribute("search", search);
        } else if (riskLevel != null && !riskLevel.isEmpty()) {
            // Filter by risk level
            predictions = predictionService.getPredictionsByRiskLevel(riskLevel, page, size);
            model.addAttribute("riskLevel", riskLevel);
        } else {
            // Get all predictions
            predictions = predictionService.getRecentPredictions(page, size);
        }

        model.addAttribute("predictions", predictions);
        model.addAttribute("currentPage", page);
        model.addAttribute("totalPages", predictions.getTotalPages());
        model.addAttribute("totalItems", predictions.getTotalElements());

        return "history";
    }

    // Dashboard
    @GetMapping("/dashboard")
    public String showDashboard(Model model) {
        StatisticsResponse stats = predictionService.getStatistics();
        model.addAttribute("stats", stats);
        model.addAttribute("apiHealthy", predictionService.isPythonApiHealthy());

        Map<String, Object> modelInfo = predictionService.getModelInfo();
        model.addAttribute("modelInfo", modelInfo);

        return "dashboard";
    }

    /**
     * View single prediction details
     */
    @GetMapping("/prediction/{id}")
    public String viewPrediction(@PathVariable Long id, Model model) {
        return predictionService.getPredictionById(id)
                .map(prediction -> {
                    model.addAttribute("prediction", prediction);
                    return "prediction-detail";
                })
                .orElse("redirect:/history");
    }

    // ========================================================================
    // REST API ENDPOINTS (JSON responses)
    // ========================================================================

    /**
     * REST API - Get all predictions
     */
    @GetMapping("/api/predictions")
    @ResponseBody
    public ResponseEntity<Page<Prediction>> getAllPredictions(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {

        Page<Prediction> predictions = predictionService.getAllPredictions(page, size);
        return ResponseEntity.ok(predictions);
    }

    /**
     * REST API - Get prediction by ID
     */
    @GetMapping("/api/predictions/{id}")
    @ResponseBody
    public ResponseEntity<Prediction> getPredictionById(@PathVariable Long id) {
        return predictionService.getPredictionById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    /**
     * REST API - Get statistics
     */
    @GetMapping("/api/statistics")
    @ResponseBody
    public ResponseEntity<StatisticsResponse> getStatistics() {
        StatisticsResponse stats = predictionService.getStatistics();
        return ResponseEntity.ok(stats);
    }

    /**
     * REST API - Health check
     */
    @GetMapping("/api/health")
    @ResponseBody
    public ResponseEntity<Map<String, Object>> healthCheck() {
        Map<String, Object> health = new HashMap<>();
        health.put("status", "UP");
        health.put("pythonApiHealthy", predictionService.isPythonApiHealthy());
        health.put("totalPredictions", predictionService.getTotalCount());
        return ResponseEntity.ok(health);
    }

    /**
     * REST API - Get model info
     */
    @GetMapping("/api/model-info")
    @ResponseBody
    public ResponseEntity<Map<String, Object>> getModelInfo() {
        Map<String, Object> modelInfo = predictionService.getModelInfo();
        return ResponseEntity.ok(modelInfo);
    }

    /**
     * REST API - Delete prediction
     */
    @DeleteMapping("/api/predictions/{id}")
    @ResponseBody
    public ResponseEntity<Map<String, String>> deletePrediction(@PathVariable Long id) {
        try {
            predictionService.deletePrediction(id);
            Map<String, String> response = new HashMap<>();
            response.put("message", "Prediction deleted successfully");
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, String> error = new HashMap<>();
            error.put("error", "Failed to delete prediction: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(error);
        }
    }

    /**
     * REST API - Submit prediction (JSON)
     */
    @PostMapping("/api/predict")
    @ResponseBody
    public ResponseEntity<PredictionResponse> predictFraudApi(
            @Valid @RequestBody FraudCheckForm form) {

        try {
            PredictionResponse response = predictionService.predictFraud(form);

            if (response.isSuccess()) {
                return ResponseEntity.ok(response);
            } else {
                return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(response);
            }

        } catch (Exception e) {
            logger.error("API prediction error: {}", e.getMessage(), e);
            PredictionResponse errorResponse = new PredictionResponse();
            errorResponse.setSuccess(false);
            errorResponse.setError(e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
        }
    }

    // ========================================================================
    // ERROR HANDLING
    // ========================================================================

    /**
     * Global exception handler
     */
    @ExceptionHandler(Exception.class)
    public String handleException(Exception e, Model model) {
        logger.error("Unexpected error: {}", e.getMessage(), e);
        model.addAttribute("error", "An unexpected error occurred: " + e.getMessage());
        return "error";
    }
}