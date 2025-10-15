package ecom.intensifier.credibilityengine.DTO;

import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PredictionResponse {
    private boolean success;

    @JsonProperty("order_id")
    private String orderId;

    private String prediction; // Trusted, Uncertain, Untrustworthy

    @JsonProperty("prediction_binary")
    private String predictionBinary; // Returned | Not Returned = 1 | 0

    private Double probability;

    @JsonProperty("risk_level")
    private String riskLevel; // LOW, MEDIUM, HIGH

    @JsonProperty("risk_score")
    private Integer riskScore; // 0-100

    private String confidence; // High, Medium, Low

    private String timestamp;

    @JsonProperty("model_version")
    private String modelVersion;

    private String error;
}
