package ecom.intensifier.credibilityengine.DTO;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;
import java.time.LocalDate;


@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FraudCheckForm {
    private String orderId;
    private String productId;
    private String userId;

    @NotNull(message = "Order date is required")
    private LocalDate orderDate;

    @NotBlank(message = "Product category is required")
    private String productCategory;

    @NotNull(message = "Product price is required")
    @DecimalMin(value = "0.01", message = "Product price must be greater than 0")
    private BigDecimal productPrice;

    @NotNull(message = "Order quantity is required")
    @Min(value = 1, message = "Order quantity must be at least 1")
    @Max(value = 10, message = "Order quantity cannot exceed 10")
    private Integer orderQuantity;

    @NotNull(message = "User age is required")
    @Min(value = 18, message = "User must be at least 18 years old")
    @Max(value = 100, message = "Invalid user age")
    private Integer userAge;

    @NotBlank(message = "User gender is required")
    private String userGender;

    @NotBlank(message = "User location is required")
    private String userLocation;

    @NotBlank(message = "Payment method is required")
    private String paymentMethod;

    @NotBlank(message = "Shipping method is required")
    private String shippingMethod;

    @NotNull(message = "Discount applied is required")
    @DecimalMin(value = "0.0", message = "Discount cannot be negative")
    private BigDecimal discountApplied;

    /**
     * Convert form to API request
     */
    public PredictionRequest toApiRequest() {
        return PredictionRequest.builder()
                .orderId(this.orderId != null && !this.orderId.isEmpty() ? this.orderId : "ORD_" + System.currentTimeMillis())
                .productId(this.productId)
                .userId(this.userId)
                .orderDate(this.orderDate.toString())
                .productCategory(this.productCategory)
                .productPrice(this.productPrice)
                .orderQuantity(this.orderQuantity)
                .userAge(this.userAge)
                .userGender(this.userGender)
                .userLocation(this.userLocation)
                .paymentMethod(this.paymentMethod)
                .shippingMethod(this.shippingMethod)
                .discountApplied(this.discountApplied)
                .build();
    }
}
