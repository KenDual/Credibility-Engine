package ecom.intensifier.credibilityengine.DTO;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class StatisticsResponse {
    private Long totalPredictions;
    private Long lowRiskCount;
    private Long mediumRiskCount;
    private Long highRiskCount;
    private Double averageRiskScore;
    private Long trustedCount;
    private Long uncertainCount;
    private Long untrustworthyCount;

    // Percentages
    private Double lowRiskPercentage;
    private Double mediumRiskPercentage;
    private Double highRiskPercentage;
}
