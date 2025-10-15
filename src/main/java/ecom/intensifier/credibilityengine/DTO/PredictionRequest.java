package ecom.intensifier.credibilityengine.DTO;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PredictionRequest {
    @JsonProperty("Order_ID")
    private String orderId;

    @JsonProperty("Product_ID")
    private String productId;

    @JsonProperty("User_ID")
    private String userId;

    @NotNull(message = "Order date is required")
    @JsonProperty("Order_Date")
    private String orderDate; // Format: YYYY-MM-DD

    @NotBlank(message = "Product category is required")
    @JsonProperty("Product_Category")
    private String productCategory;

    @NotNull(message = "Product price is required")
    @DecimalMin(value = "0.0", message = "Product price must be positive")
    @JsonProperty("Product_Price")
    private BigDecimal productPrice;

    @NotNull(message = "Order quantity is required")
    @Min(value = 1, message = "Order quantity must be at least 1")
    @JsonProperty("Order_Quantity")
    private Integer orderQuantity;

    @NotNull(message = "User age is required")
    @Min(value = 18, message = "User age must be at least 18")
    @Max(value = 120, message = "User age must be less than 120")
    @JsonProperty("User_Age")
    private Integer userAge;

    @NotBlank(message = "User gender is required")
    @JsonProperty("User_Gender")
    private String userGender;

    @NotBlank(message = "User location is required")
    @JsonProperty("User_Location")
    private String userLocation;

    @NotBlank(message = "Payment method is required")
    @JsonProperty("Payment_Method")
    private String paymentMethod;

    @NotBlank(message = "Shipping method is required")
    @JsonProperty("Shipping_Method")
    private String shippingMethod;

    @NotNull(message = "Discount applied is required")
    @DecimalMin(value = "0.0", message = "Discount must be positive")
    @JsonProperty("Discount_Applied")
    private BigDecimal discountApplied;
}
