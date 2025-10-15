package ecom.intensifier.credibilityengine.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

/**
 * JPA Entity for storing fraud detection predictions
 */
@Entity
@Table(name = "predictions", indexes = {
        @Index(name = "idx_order_id", columnList = "order_id"),
        @Index(name = "idx_user_id", columnList = "user_id"),
        @Index(name = "idx_risk_level", columnList = "risk_level"),
        @Index(name = "idx_predicted_at", columnList = "predicted_at")
})
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Prediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    // Order Information
    @Column(name = "order_id", length = 50)
    private String orderId;

    @Column(name = "product_id", length = 50)
    private String productId;

    @Column(name = "user_id", length = 50)
    private String userId;

    // Prediction Results
    @Column(name = "prediction_result", nullable = false, length = 20)
    private String predictionResult; // Trusted, Uncertain, Untrustworthy

    @Column(name = "prediction_binary", nullable = false, length = 20)
    private String predictionBinary; // Returned, Not Returned

    @Column(name = "probability", nullable = false, precision = 10, scale = 6)
    private BigDecimal probability;

    @Column(name = "risk_level", nullable = false, length = 10)
    private String riskLevel; // LOW, MEDIUM, HIGH

    @Column(name = "risk_score", nullable = false)
    private Integer riskScore; // 0-100

    @Column(name = "confidence", length = 10)
    private String confidence; // High, Medium, Low

    // Input Features (Order Details)
    @Column(name = "product_category", length = 50)
    private String productCategory;

    @Column(name = "product_price", precision = 10, scale = 2)
    private BigDecimal productPrice;

    @Column(name = "order_quantity")
    private Integer orderQuantity;

    @Column(name = "discount_applied", precision = 10, scale = 2)
    private BigDecimal discountApplied;

    @Column(name = "discount_percentage", precision = 10, scale = 2)
    private BigDecimal discountPercentage;

    @Column(name = "total_order_value", precision = 10, scale = 2)
    private BigDecimal totalOrderValue;

    // User Information
    @Column(name = "user_age")
    private Integer userAge;

    @Column(name = "user_gender", length = 10)
    private String userGender;

    @Column(name = "user_location", length = 100)
    private String userLocation;

    // Transaction Details
    @Column(name = "payment_method", length = 30)
    private String paymentMethod;

    @Column(name = "shipping_method", length = 30)
    private String shippingMethod;

    @Column(name = "order_date")
    private LocalDate orderDate;

    // Metadata
    @Column(name = "model_version", length = 20)
    private String modelVersion;

    @Column(name = "predicted_at", nullable = false)
    private LocalDateTime predictedAt;

    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        if (predictedAt == null) {
            predictedAt = LocalDateTime.now();
        }
    }

    /**
     * Get risk level color for UI
     */
    @Transient
    public String getRiskLevelColor() {
        if (riskLevel == null)
            return "secondary";

        switch (riskLevel) {
            case "LOW":
                return "success"; // Green
            case "MEDIUM":
                return "warning"; // Yellow
            case "HIGH":
                return "danger"; // Red
            default:
                return "secondary";
        }
    }

    /**
     * Get prediction result badge color
     */
    @Transient
    public String getPredictionBadgeColor() {
        if (predictionResult == null)
            return "secondary";

        switch (predictionResult) {
            case "Trusted":
                return "success";
            case "Uncertain":
                return "warning";
            case "Untrustworthy":
                return "danger";
            default:
                return "secondary";
        }
    }
}