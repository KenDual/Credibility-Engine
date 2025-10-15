package ecom.intensifier.credibilityengine.repository;

import ecom.intensifier.credibilityengine.entity.Prediction;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

/**
 * Repository interface for Prediction entity
 * Provides database access methods for predictions
 */
@Repository
public interface PredictionRepository extends JpaRepository<Prediction, Long> {

    /**
     * Find prediction by order ID
     */
    Optional<Prediction> findByOrderId(String orderId);

    /**
     * Find all predictions by user ID
     */
    List<Prediction> findByUserId(String userId);

    /**
     * Find all predictions by risk level
     */
    List<Prediction> findByRiskLevel(String riskLevel);

    /**
     * Find predictions by risk level with pagination
     */
    Page<Prediction> findByRiskLevel(String riskLevel, Pageable pageable);

    /**
     * Find all predictions ordered by predicted date (newest first)
     */
    Page<Prediction> findAllByOrderByPredictedAtDesc(Pageable pageable);

    /**
     * Find predictions within date range
     */
    @Query("SELECT p FROM Prediction p WHERE p.predictedAt BETWEEN :startDate AND :endDate ORDER BY p.predictedAt DESC")
    List<Prediction> findPredictionsBetweenDates(
            @Param("startDate") LocalDateTime startDate,
            @Param("endDate") LocalDateTime endDate
    );

    /**
     * Count predictions by risk level
     */
    Long countByRiskLevel(String riskLevel);

    /**
     * Count predictions by prediction result
     */
    Long countByPredictionResult(String predictionResult);

    /**
     * Get total count of predictions
     */
    @Query("SELECT COUNT(p) FROM Prediction p")
    Long getTotalCount();

    /**
     * Get average risk score
     */
    @Query("SELECT AVG(p.riskScore) FROM Prediction p")
    Double getAverageRiskScore();

    /**
     * Get predictions with high risk (score > 70)
     */
    @Query("SELECT p FROM Prediction p WHERE p.riskScore > 70 ORDER BY p.riskScore DESC")
    List<Prediction> findHighRiskPredictions();

    /**
     * Get recent high-risk predictions with pagination
     */
    @Query("SELECT p FROM Prediction p WHERE p.riskLevel = 'HIGH' ORDER BY p.predictedAt DESC")
    Page<Prediction> findRecentHighRiskPredictions(Pageable pageable);

    /**
     * Search predictions by order ID or user ID
     */
    @Query("SELECT p FROM Prediction p WHERE p.orderId LIKE %:searchTerm% OR p.userId LIKE %:searchTerm%")
    Page<Prediction> searchByOrderOrUserId(@Param("searchTerm") String searchTerm, Pageable pageable);

    /**
     * Find predictions by product category
     */
    List<Prediction> findByProductCategory(String productCategory);

    /**
     * Get statistics for dashboard
     */
    @Query("SELECT " +
           "COUNT(p), " +
           "SUM(CASE WHEN p.riskLevel = 'LOW' THEN 1 ELSE 0 END), " +
           "SUM(CASE WHEN p.riskLevel = 'MEDIUM' THEN 1 ELSE 0 END), " +
           "SUM(CASE WHEN p.riskLevel = 'HIGH' THEN 1 ELSE 0 END), " +
           "AVG(p.riskScore) " +
           "FROM Prediction p")
    Object[] getStatistics();

    /**
     * Delete old predictions (older than specified date)
     */
    void deleteByPredictedAtBefore(LocalDateTime dateTime);

    /**
     * Find predictions by user location
     */
    List<Prediction> findByUserLocation(String location);

    /**
     * Get predictions grouped by risk level with counts
     */
    @Query("SELECT p.riskLevel, COUNT(p) FROM Prediction p GROUP BY p.riskLevel")
    List<Object[]> countByRiskLevelGrouped();

    /**
     * Find recent predictions (last N days)
     */
    @Query("SELECT p FROM Prediction p WHERE p.predictedAt >= :sinceDate ORDER BY p.predictedAt DESC")
    List<Prediction> findRecentPredictions(@Param("sinceDate") LocalDateTime sinceDate);
}