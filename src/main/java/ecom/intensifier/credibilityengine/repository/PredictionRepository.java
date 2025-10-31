package ecom.intensifier.credibilityengine.repository;

import ecom.intensifier.credibilityengine.DTO.CategoryStatsDTO;
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

@Repository
public interface PredictionRepository extends JpaRepository<Prediction, Long> {

    Optional<Prediction> findByOrderId(String orderId);

    List<Prediction> findByUserId(String userId);

    List<Prediction> findByRiskLevel(String riskLevel);

    Page<Prediction> findByRiskLevel(String riskLevel, Pageable pageable);

    Page<Prediction> findAllByOrderByPredictedAtDesc(Pageable pageable);

    @Query("SELECT p FROM Prediction p WHERE p.predictedAt BETWEEN :startDate AND :endDate ORDER BY p.predictedAt DESC")
    List<Prediction> findPredictionsBetweenDates(
            @Param("startDate") LocalDateTime startDate,
            @Param("endDate") LocalDateTime endDate);

    Long countByRiskLevel(String riskLevel);

    Long countByPredictionResult(String predictionResult);

    @Query("SELECT COUNT(p) FROM Prediction p")
    Long getTotalCount();

    @Query("SELECT AVG(p.riskScore) FROM Prediction p")
    Double getAverageRiskScore();

    @Query("SELECT p FROM Prediction p WHERE p.riskScore > 70 ORDER BY p.riskScore DESC")
    List<Prediction> findHighRiskPredictions();

    @Query("SELECT p FROM Prediction p WHERE p.riskLevel = 'HIGH' ORDER BY p.predictedAt DESC")
    Page<Prediction> findRecentHighRiskPredictions(Pageable pageable);

    @Query("SELECT p FROM Prediction p WHERE p.orderId LIKE %:searchTerm% OR p.userId LIKE %:searchTerm%")
    Page<Prediction> searchByOrderOrUserId(@Param("searchTerm") String searchTerm, Pageable pageable);

    List<Prediction> findByProductCategory(String productCategory);

    @Query("SELECT " +
            "COUNT(p), " +
            "SUM(CASE WHEN p.riskLevel = 'LOW' THEN 1 ELSE 0 END), " +
            "SUM(CASE WHEN p.riskLevel = 'MEDIUM' THEN 1 ELSE 0 END), " +
            "SUM(CASE WHEN p.riskLevel = 'HIGH' THEN 1 ELSE 0 END), " +
            "AVG(p.riskScore) " +
            "FROM Prediction p")
    Object[] getStatistics();

    void deleteByPredictedAtBefore(LocalDateTime dateTime);

    List<Prediction> findByUserLocation(String location);

    @Query("SELECT p.riskLevel, COUNT(p) FROM Prediction p GROUP BY p.riskLevel")
    List<Object[]> countByRiskLevelGrouped();

    @Query("SELECT p FROM Prediction p WHERE p.predictedAt >= :sinceDate ORDER BY p.predictedAt DESC")
    List<Prediction> findRecentPredictions(@Param("sinceDate") LocalDateTime sinceDate);

    @Query("SELECT new ecom.intensifier.credibilityengine.DTO.CategoryStatsDTO(" +
            "p.productCategory, " +
            "COUNT(p), " +
            "SUM(CASE WHEN p.riskLevel = 'LOW' THEN 1 ELSE 0 END), " +
            "SUM(CASE WHEN p.riskLevel = 'MEDIUM' THEN 1 ELSE 0 END), " +
            "SUM(CASE WHEN p.riskLevel = 'HIGH' THEN 1 ELSE 0 END), " +
            "AVG(p.probability)) " +
            "FROM Prediction p " +
            "GROUP BY p.productCategory")
    List<CategoryStatsDTO> getCategoryStatistics();
}