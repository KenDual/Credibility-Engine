package ecom.intensifier.credibilityengine.DTO;

public class CategoryStatsDTO {
    private String category;
    private Long total;
    private Long lowRisk;
    private Long mediumRisk;
    private Long highRisk;
    private Double avgProbability;

    // Constructors
    public CategoryStatsDTO() {
    }

    public CategoryStatsDTO(String category, Long total, Long lowRisk,
            Long mediumRisk, Long highRisk, Double avgProbability) {
        this.category = category;
        this.total = total;
        this.lowRisk = lowRisk;
        this.mediumRisk = mediumRisk;
        this.highRisk = highRisk;
        this.avgProbability = avgProbability;
    }

    // Getters and Setters
    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    public Long getTotal() {
        return total;
    }

    public void setTotal(Long total) {
        this.total = total;
    }

    public Long getLowRisk() {
        return lowRisk;
    }

    public void setLowRisk(Long lowRisk) {
        this.lowRisk = lowRisk;
    }

    public Long getMediumRisk() {
        return mediumRisk;
    }

    public void setMediumRisk(Long mediumRisk) {
        this.mediumRisk = mediumRisk;
    }

    public Long getHighRisk() {
        return highRisk;
    }

    public void setHighRisk(Long highRisk) {
        this.highRisk = highRisk;
    }

    public Double getAvgProbability() {
        return avgProbability;
    }

    public void setAvgProbability(Double avgProbability) {
        this.avgProbability = avgProbability;
    }
}