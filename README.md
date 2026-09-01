# 🚚 APL Logistics – Delivery Intelligence

A data-driven logistics analytics and decision-support dashboard designed to analyze delivery performance, identify delay risks, and provide insights into logistics efficiency.

The project uses logistics shipment data to transform raw operational records into interactive visualizations, delivery classifications, delay-risk analysis, and actionable business insights.

## 🌐 Live Application

The project is deployed using Streamlit and provides an interactive dashboard for exploring logistics delivery performance.

**Live Demo:** Replace this line with the deployed Streamlit app link.

---

## 📌 Problem Statement

Logistics and last-mile delivery operations generate large volumes of shipment data. Analyzing this data manually makes it difficult to identify delivery delays, understand operational patterns, compare shipping modes and regions, and recognize factors associated with delivery risk.

This project addresses the problem by developing an interactive logistics intelligence dashboard that analyzes shipment data and presents meaningful operational insights through data visualization and filtering.

---

## 🎯 Objectives

- Analyze overall delivery performance.
- Identify delayed and on-time shipments.
- Calculate the difference between actual and scheduled shipping time.
- Classify shipments based on delivery performance.
- Analyze late-delivery risk.
- Compare different shipping modes.
- Analyze delivery performance across markets and regions.
- Study customer-segment patterns.
- Provide interactive filtering for operational analysis.
- Generate data-driven logistics insights and recommendations.

---

## ⭐ Key Features

### 📊 Delivery Performance Analysis
- Delivery status distribution
- Delivery classification
- Delay-gap analysis
- Shipment-level performance indicators

### ⚠️ Delay Risk Analysis
- Identification of late-delivery risk
- Actual vs scheduled shipping comparison
- Delay classification
- Risk-oriented operational insights

### 🚚 Shipping Mode Analysis
- Comparison of delivery performance across shipping modes
- Identification of patterns in shipment delays
- Interactive shipping-mode filtering

### 🌎 Regional & Market Analysis
- Market-level analysis
- Regional delivery performance
- Comparison of logistics patterns across geographical areas

### 👥 Customer & Product Analysis
- Customer-segment analysis
- Product/category-level insights
- Shipment and customer patterns

### 🔎 Interactive Dashboard Controls
Users can explore the data using filters such as:

- Shipping Mode
- Market
- Order Region
- Customer Segment
- Date Range

---

## 🧮 Delay Calculation

The project derives a **Delay Gap** using the difference between actual shipping time and scheduled shipping time.

```text
Delay Gap = Actual Shipping Days − Scheduled Shipping Days
