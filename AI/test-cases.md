# Test Data cho Form - Credibility Engine

## Test Case 1: LOW RISK - Trusted Customer ✅
**Mục đích:** Test case lý tưởng - khách hàng đáng tin cậy

```
Order ID: ORD_LOW_001
Product ID: PROD_BOOK_123
User ID: USER_TRUSTED_01
Order Date: 2025-10-15
Product Category: Books
Product Price: 29.99
Order Quantity: 1
Discount Applied: 3.00
User Age: 45
User Gender: Male
User Location: City1
Payment Method: Credit Card
Shipping Method: Standard
```
**Expected:** LOW risk (~20-30%), "Trusted"

---

## Test Case 2: MEDIUM RISK - Uncertain ⚠️
**Mục đích:** Trường hợp không chắc chắn

```
Order ID: ORD_MED_001
Product ID: PROD_CLOTH_456
User ID: USER_NORMAL_02
Order Date: 2025-10-20
Product Category: Clothing
Product Price: 199.99
Order Quantity: 3
Discount Applied: 40.00
User Age: 28
User Gender: Female
User Location: City25
Payment Method: PayPal
Shipping Method: Express
```
**Expected:** MEDIUM risk (~40-60%), "Uncertain"

---

## Test Case 3: HIGH RISK - Untrustworthy 🚨
**Mục đích:** Khách hàng đáng ngờ

```
Order ID: ORD_HIGH_001
Product ID: PROD_ELEC_789
User ID: USER_RISKY_03
Order Date: 2025-10-25
Product Category: Electronics
Product Price: 999.99
Order Quantity: 5
Discount Applied: 250.00
User Age: 21
User Gender: Male
User Location: City99
Payment Method: Cash
Shipping Method: Next-Day
```
**Expected:** HIGH risk (~70-90%), "Untrustworthy"

---

## Test Case 4: Electronics - High Value Order 💻
**Mục đích:** Test với giá trị cao

```
Order ID: ORD_ELEC_002
Product ID: PROD_LAPTOP_001
User ID: USER_TECH_04
Order Date: 2025-10-18
Product Category: Electronics
Product Price: 1299.00
Order Quantity: 2
Discount Applied: 150.00
User Age: 35
User Gender: Female
User Location: City10
Payment Method: Credit Card
Shipping Method: Express
```
**Expected:** MEDIUM-HIGH risk

---

## Test Case 5: Toys - Low Value, Multiple Items 🧸
**Mục đích:** Test với sản phẩm giá thấp nhưng số lượng nhiều

```
Order ID: ORD_TOY_001
Product ID: PROD_TOY_555
User ID: USER_PARENT_05
Order Date: 2025-10-12
Product Category: Toys
Product Price: 15.50
Order Quantity: 8
Discount Applied: 10.00
User Age: 38
User Gender: Female
User Location: City5
Payment Method: Debit Card
Shipping Method: Standard
```
**Expected:** LOW-MEDIUM risk

---

## Test Case 6: Home - First Time Buyer 🏠
**Mục đích:** Khách hàng mới, đơn hàng bình thường

```
Order ID: ORD_HOME_001
Product ID: PROD_FURNITURE_100
User ID: USER_NEW_06
Order Date: 2025-10-22
Product Category: Home
Product Price: 450.00
Order Quantity: 1
Discount Applied: 45.00
User Age: 42
User Gender: Male
User Location: City15
Payment Method: Credit Card
Shipping Method: Standard
```
**Expected:** LOW-MEDIUM risk

---

## Test Case 7: Edge Case - Minimum Values 📉
**Mục đích:** Test với giá trị tối thiểu

```
Order ID: ORD_MIN_001
Product ID: PROD_MIN_001
User ID: USER_MIN_07
Order Date: 2025-10-10
Product Category: Books
Product Price: 5.00
Order Quantity: 1
Discount Applied: 0.00
User Age: 18
User Gender: Female
User Location: City1
Payment Method: Credit Card
Shipping Method: Standard
```
**Expected:** LOW risk

---

## Test Case 8: Edge Case - Maximum Values 📈
**Mục đích:** Test với giá trị tối đa

```
Order ID: ORD_MAX_001
Product ID: PROD_MAX_999
User ID: USER_MAX_08
Order Date: 2025-10-28
Product Category: Electronics
Product Price: 2999.99
Order Quantity: 10
Discount Applied: 500.00
User Age: 100
User Gender: Male
User Location: City100
Payment Method: Cash
Shipping Method: Next-Day
```
**Expected:** HIGH risk

---

## Test Case 9: Cash Payment - High Risk Indicator 💵
**Mục đích:** Test với Cash on Delivery (risk factor)

```
Order ID: ORD_CASH_001
Product ID: PROD_CASH_200
User ID: USER_CASH_09
Order Date: 2025-10-24
Product Category: Clothing
Product Price: 599.00
Order Quantity: 4
Discount Applied: 120.00
User Age: 25
User Gender: Male
User Location: City50
Payment Method: Cash
Shipping Method: Next-Day
```
**Expected:** MEDIUM-HIGH risk

---

## Test Case 10: High Discount Percentage 🎁
**Mục đích:** Test với discount cao (risk indicator)

```
Order ID: ORD_DISC_001
Product ID: PROD_DISC_300
User ID: USER_DISC_10
Order Date: 2025-10-26
Product Category: Electronics
Product Price: 500.00
Order Quantity: 3
Discount Applied: 400.00
User Age: 30
User Gender: Female
User Location: City40
Payment Method: Debit Card
Shipping Method: Express
```
**Expected:** HIGH risk (discount > 80%)

---

## Test Matrix Summary

| Case | Category | Price | Qty | Age | Payment | Expected Risk |
|------|----------|-------|-----|-----|---------|---------------|
| 1 | Books | 29.99 | 1 | 45 | Credit | LOW ✅ |
| 2 | Clothing | 199.99 | 3 | 28 | PayPal | MEDIUM ⚠️ |
| 3 | Electronics | 999.99 | 5 | 21 | Cash | HIGH 🚨 |
| 4 | Electronics | 1299.00 | 2 | 35 | Credit | MEDIUM-HIGH |
| 5 | Toys | 15.50 | 8 | 38 | Debit | LOW-MEDIUM |
| 6 | Home | 450.00 | 1 | 42 | Credit | LOW-MEDIUM |
| 7 | Books | 5.00 | 1 | 18 | Credit | LOW |
| 8 | Electronics | 2999.99 | 10 | 100 | Cash | HIGH 🚨 |
| 9 | Clothing | 599.00 | 4 | 25 | Cash | MEDIUM-HIGH |
| 10 | Electronics | 500.00 | 3 | 30 | Debit | HIGH 🚨 |

---

## Test Execution Checklist

Cho mỗi test case:
- [ ] Form nhận đầy đủ data
- [ ] Submit thành công
- [ ] Python API nhận request
- [ ] Result page hiển thị
- [ ] Risk level đúng với expected
- [ ] Data lưu vào database
- [ ] History page hiển thị record