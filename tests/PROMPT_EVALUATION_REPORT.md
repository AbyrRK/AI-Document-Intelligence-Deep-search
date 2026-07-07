# Prompt Engineering Evaluation Report
**Project:** PFE2026  
**Date:** 2026-05-19T15:10:38.306267  
**LLM Backend:** ollama  
**Session ID:** dbdb7501-cc3f-4929-b003-86f78e3f03c3  

## Executive Summary

**Winner:** `B  grounded + cited` with an average score of **87%**.

## Leaderboard

| Variant | Avg Score | Case Scores |
|---------|-----------|-------------|
| B  grounded + cited | 87% | 100% 100% 80% 80% 80% 80% 100% 80% 100% 100% 80% 80% 90% 70% 80% 60% 100% 100% 100% |
| A  baseline (current) | 85% | 100% 100% 100% 80% 80% 80% 100% 80% 100% 80% 80% 80% 90% 70% 80% 60% 100% 80% 80% |
| C  chain-of-thought | 85% | 100% 100% 80% 80% 80% 80% 100% 80% 100% 80% 80% 80% 90% 70% 80% 60% 100% 80% 100% |
| D  multilingual + cited | 85% | 100% 100% 80% 80% 100% 80% 100% 80% 100% 80% 80% 80% 90% 60% 80% 60% 80% 100% 80% |

## Detailed Results

### doc_summary: General document understanding
**Question:** What is this document about?

| Rank | Variant | Score | Status | Answer Preview |
|------|---------|-------|--------|----------------|
| 1 | A  baseline (current) | 100% | ok | This document describes the development of an application to generate OLAP cubes for ACTEOL's clients, specifically a sp… |
| 2 | B  grounded + cited | 100% | ok | This document describes the development of an application to generate a cube OLAP for clients of ACTEOL, a company speci… |
| 3 | C  chain-of-thought | 100% | ok | This document describes the development of an application to generate cube OLAP for clients of ACTEOL, a company providi… |
| 4 | D  multilingual + cited | 100% | ok | This document describes a project for developing an application that helps manage data for the company ACTEOL. The appli… |

### doc_title: Extract document title
**Question:** What is the title of this document?

| Rank | Variant | Score | Status | Answer Preview |
|------|---------|-------|--------|----------------|
| 1 | A  baseline (current) | 100% | ok | The title of the document is "**Analyse et spécification des besoins, Architecture générale et Diagrammes de séquences o… |
| 2 | B  grounded + cited | 100% | ok | [Excerpt 2] states that the document's title is "Table des matières".  Therefore, the title of the document is **"Table … |
| 3 | C  chain-of-thought | 100% | ok | The title of the document is "**Analyse et spécification des besoins**".   This is based on the table of contents provid… |
| 4 | D  multilingual + cited | 100% | ok | The title of this document is **"Analyse et spécification des besoins"**.   |

### detailed_explanation: Ask for details with evidence
**Question:** Can you give me more details and show me where you found this?

| Rank | Variant | Score | Status | Answer Preview |
|------|---------|-------|--------|----------------|
| 1 | A  baseline (current) | 100% | ok | The document describes the application's visualization of client interactions with offers, specifically mentioning two e… |
| 2 | B  grounded + cited | 80% | ok | Please provide me with the specific information you'd like to know more about. I can then cite the relevant source withi… |
| 3 | C  chain-of-thought | 80% | ok | I can definitely provide more details about the information in the document!   Here's a breakdown of what I've extracted… |
| 4 | D  multilingual + cited | 80% | ok | Please provide the question you would like answered! I will then do my best to answer it, using the provided context, an… |

### budget_extraction: Extract budget or financial amount
**Question:** What is the exact budget mentioned in the document?

| Rank | Variant | Score | Status | Answer Preview |
|------|---------|-------|--------|----------------|
| 1 | A  baseline (current) | 80% | ok | The provided text does not mention any budget.   |
| 2 | B  grounded + cited | 80% | ok | Not found in document.   |
| 3 | C  chain-of-thought | 80% | ok | Not found in document.   |
| 4 | D  multilingual + cited | 80% | ok | The provided text does not mention any budget.   |

### total_amount_due: Extract total payment amount
**Question:** What is the total amount due?

| Rank | Variant | Score | Status | Answer Preview |
|------|---------|-------|--------|----------------|
| 1 | D  multilingual + cited | 100% | ok | [Excerpt 4]  Table des Figures 6.10 Interface de génération de cube OLAP 51   Please note that I cannot provide a specif… |
| 2 | A  baseline (current) | 80% | ok | This document does not contain information about a total amount due.   |
| 3 | B  grounded + cited | 80% | ok | Not found in document.   |
| 4 | C  chain-of-thought | 80% | ok | Not found in document.   |

### deadline_extraction: Extract deadline or due date
**Question:** What is the due date or deadline mentioned?

| Rank | Variant | Score | Status | Answer Preview |
|------|---------|-------|--------|----------------|
| 1 | A  baseline (current) | 80% | ok | This document does not mention a due date or deadline.   |
| 2 | B  grounded + cited | 80% | ok | Not found in document.   |
| 3 | C  chain-of-thought | 80% | ok | Not found in document.   |
| 4 | D  multilingual + cited | 80% | ok | There is no mention of a due date or deadline in the provided text.   |

### company_names: Extract organizations or companies
**Question:** What company names are mentioned in the document?

| Rank | Variant | Score | Status | Answer Preview |
|------|---------|-------|--------|----------------|
| 1 | A  baseline (current) | 100% | ok | The following company names are mentioned in the document:  * **Acteol**  * **Esprit**  * **British Train Company** (imp… |
| 2 | B  grounded + cited | 100% | ok | Here are the company names mentioned in the document:  * **Acteol**  * **Esprit**   * **British Train Company** (implied… |
| 3 | C  chain-of-thought | 100% | ok | Here are the company names mentioned in the document:  * **Acteol**  * **Esprit**   * **British Train Company** (implied… |
| 4 | D  multilingual + cited | 100% | ok | Here are the company names mentioned in the document:  * **Acteol**  * **Esprit**  * **British Train Company** (implied,… |

### important_dates: Extract important dates
**Question:** What important dates are mentioned in this document?

| Rank | Variant | Score | Status | Answer Preview |
|------|---------|-------|--------|----------------|
| 1 | A  baseline (current) | 80% | ok | This document does not mention any specific dates.   |
| 2 | B  grounded + cited | 80% | ok | Not found in document.   |
| 3 | C  chain-of-thought | 80% | ok | Not found in document.   |
| 4 | D  multilingual + cited | 80% | ok | There are no specific dates mentioned in this document.   |

### key_points: Summarize key information
**Question:** What are the key points mentioned in this document?

| Rank | Variant | Score | Status | Answer Preview |
|------|---------|-------|--------|----------------|
| 1 | A  baseline (current) | 100% | ok | Here are the key points mentioned in the document:  * **Project:** Develop a Windows Forms application to facilitate cub… |
| 2 | B  grounded + cited | 100% | ok | Here are the key points mentioned in the document:  * **Project Objective:** Develop a Windows Forms application to faci… |
| 3 | C  chain-of-thought | 100% | ok | Here are some key points from the document:  * **Project Goal:** The document describes a project undertaken by an engin… |
| 4 | D  multilingual + cited | 100% | ok | Here are the key points mentioned in the document:  **Project Overview:**  * **Purpose:** Develop a Windows Forms applic… |

### required_actions: Identify requested actions
**Question:** Are there any actions or tasks required in this document?

| Rank | Variant | Score | Status | Answer Preview |
|------|---------|-------|--------|----------------|
| 1 | B  grounded + cited | 100% | ok | [Excerpt 2] states: "Ce chapitre consiste à comprendre le contexte du système. Il s'agit de déterminer les fonctionnalit… |
| 2 | A  baseline (current) | 80% | ok | Yes, the document describes several actions and tasks.   Here are some examples:  * **Identify actors:**  Understanding … |
| 3 | C  chain-of-thought | 80% | ok | Yes, the document describes several actions and tasks. Here are some examples:  * **Identify actors:**  The document out… |
| 4 | D  multilingual + cited | 80% | ok | Yes, the document describes several actions and tasks. Here are some examples:  * **Identify actors:** The document outl… |

### payment_terms: Extract payment conditions
**Question:** What payment terms are specified?

| Rank | Variant | Score | Status | Answer Preview |
|------|---------|-------|--------|----------------|
| 1 | A  baseline (current) | 80% | ok | This document does not contain information about payment terms.   |
| 2 | B  grounded + cited | 80% | ok | Not found in document.   |
| 3 | C  chain-of-thought | 80% | ok | Not found in document.   |
| 4 | D  multilingual + cited | 80% | ok | Please provide me with the context or document you would like me to analyze for payment terms.    |

### contract_duration: Extract duration information
**Question:** What is the contract duration mentioned in the document?

| Rank | Variant | Score | Status | Answer Preview |
|------|---------|-------|--------|----------------|
| 1 | A  baseline (current) | 80% | ok | This document does not mention the contract duration.   |
| 2 | B  grounded + cited | 80% | ok | Not found in document.   |
| 3 | C  chain-of-thought | 80% | ok | Not found in document.   |
| 4 | D  multilingual + cited | 80% | ok | The provided text does not mention the contract duration.   |

### french_summary: French multilingual test
**Question:** Quel est le sujet principal de ce document ?

| Rank | Variant | Score | Status | Answer Preview |
|------|---------|-------|--------|----------------|
| 1 | A  baseline (current) | 90% | ok | The main subject of this document is the development of a decision-making system based on data from a sports facility to… |
| 2 | B  grounded + cited | 90% | ok | The main subject of this document is the development of a software application to help sports facilities make better dec… |
| 3 | C  chain-of-thought | 90% | ok | The main subject of this document is the development of a decision-support system for a sports facility using data analy… |
| 4 | D  multilingual + cited | 90% | ok | The main subject of this document is the development of a decision-making system based on data from a sports facility to… |

### french_deadline: French deadline extraction
**Question:** Quelle est la date limite mentionnée dans le document ?

| Rank | Variant | Score | Status | Answer Preview |
|------|---------|-------|--------|----------------|
| 1 | A  baseline (current) | 70% | ok | This document does not contain a date limit.   |
| 2 | B  grounded + cited | 70% | ok | Not found in document.   |
| 3 | C  chain-of-thought | 70% | ok | Not found in document.   |
| 4 | D  multilingual + cited | 60% | ok | The provided text does not contain a specific deadline mentioned.    Please let me know if you have other questions!   |

### arabic_summary: Arabic multilingual test
**Question:** ما هو موضوع هذا المستند؟

| Rank | Variant | Score | Status | Answer Preview |
|------|---------|-------|--------|----------------|
| 1 | A  baseline (current) | 80% | ok | This document is about developing an application to generate OLAP cubes for ACTEOL's clients, using Windows Forms and C#… |
| 2 | B  grounded + cited | 80% | ok | [Excerpt 3] states that this work is about developing an application in Windows forms to facilitate the generation of OL… |
| 3 | C  chain-of-thought | 80% | ok | The document describes the development of an application that helps manage accounts and generates data-driven dashboards… |
| 4 | D  multilingual + cited | 80% | ok | This document is about the development of an application that helps account managers at ACTEOL generate a cube OLAP and … |

### arabic_amount: Arabic amount extraction
**Question:** ما هو المبلغ الإجمالي المستحق المذكور في المستند؟

| Rank | Variant | Score | Status | Answer Preview |
|------|---------|-------|--------|----------------|
| 1 | A  baseline (current) | 60% | ok | I don't know the answer to that question.   |
| 2 | B  grounded + cited | 60% | ok | Not found in document.   |
| 3 | C  chain-of-thought | 60% | ok | Not found in document.   |
| 4 | D  multilingual + cited | 60% | ok | I am sorry, but the provided context does not contain information about the total amount owed.    Please let me know if … |

### out_of_scope_ceo: Out-of-scope refusal test
**Question:** What is the CEO's favorite color?

| Rank | Variant | Score | Status | Answer Preview |
|------|---------|-------|--------|----------------|
| 1 | A  baseline (current) | 100% | ok | I don't know.   |
| 2 | B  grounded + cited | 100% | ok | Not found in document.   |
| 3 | C  chain-of-thought | 100% | ok | Not found in document.   |
| 4 | D  multilingual + cited | 80% | ok | I am sorry, but I cannot answer that question. The provided text does not contain information about the CEO's favorite c… |

### out_of_scope_prediction: Unsupported prediction request
**Question:** Can you predict the company's revenue next year?

| Rank | Variant | Score | Status | Answer Preview |
|------|---------|-------|--------|----------------|
| 1 | B  grounded + cited | 100% | ok | Not found in document.   |
| 2 | D  multilingual + cited | 100% | ok | I cannot predict the company's revenue for next year.   My knowledge is based on the provided text, which focuses on Act… |
| 3 | A  baseline (current) | 80% | ok | I cannot predict Acteol's revenue next year.   Here's why:  * **Lack of Financial Data:** The provided text focuses on A… |
| 4 | C  chain-of-thought | 80% | ok | I cannot predict the company's revenue for next year.   Here's why:  * **Lack of Financial Data:** The provided excerpts… |

### ambiguous_question: Ambiguous comparison question
**Question:** Is this company better than its competitors?

| Rank | Variant | Score | Status | Answer Preview |
|------|---------|-------|--------|----------------|
| 1 | B  grounded + cited | 100% | ok | Not found in document.   |
| 2 | C  chain-of-thought | 100% | ok | Not found in document.   |
| 3 | A  baseline (current) | 80% | ok | The provided text discusses Acteol's role in collecting customer data and providing a "powerful view client" but doesn't… |
| 4 | D  multilingual + cited | 80% | ok | [Excerpt 3] states that Acteol is specialized in creating CRM (Customer Relationship Management) tools to improve commun… |

## Regression Assertions
The following assertions must pass for any prompt change to be accepted:

- `doc_summary` → `contains` must be ≥ 100%
- `doc_summary` → `citation` must be ≥ 100%
- `doc_summary` → `refusal` must be ≥ 100%
- `doc_summary` → `language` must be ≥ 100%
- `doc_title` → `contains` must be ≥ 100%
- `doc_title` → `citation` must be ≥ 100%
- `doc_title` → `refusal` must be ≥ 100%
- `doc_title` → `language` must be ≥ 100%
- `detailed_explanation` → `contains` must be ≥ 100%
- `detailed_explanation` → `citation` must be ≥ 100%
- `detailed_explanation` → `refusal` must be ≥ 100%
- `detailed_explanation` → `language` must be ≥ 100%
- `budget_extraction` → `contains` must be ≥ 100%
- `budget_extraction` → `refusal` must be ≥ 100%
- `budget_extraction` → `language` must be ≥ 100%
- `total_amount_due` → `contains` must be ≥ 100%
- `total_amount_due` → `citation` must be ≥ 100%
- `total_amount_due` → `refusal` must be ≥ 100%
- `total_amount_due` → `language` must be ≥ 100%
- `deadline_extraction` → `contains` must be ≥ 100%
- `deadline_extraction` → `refusal` must be ≥ 100%
- `deadline_extraction` → `language` must be ≥ 100%
- `company_names` → `contains` must be ≥ 100%
- `company_names` → `citation` must be ≥ 100%
- `company_names` → `refusal` must be ≥ 100%
- `company_names` → `language` must be ≥ 100%
- `important_dates` → `contains` must be ≥ 100%
- `important_dates` → `refusal` must be ≥ 100%
- `important_dates` → `language` must be ≥ 100%
- `key_points` → `contains` must be ≥ 100%
- `key_points` → `citation` must be ≥ 100%
- `key_points` → `refusal` must be ≥ 100%
- `key_points` → `language` must be ≥ 100%
- `required_actions` → `contains` must be ≥ 100%
- `required_actions` → `citation` must be ≥ 100%
- `required_actions` → `refusal` must be ≥ 100%
- `required_actions` → `language` must be ≥ 100%
- `payment_terms` → `contains` must be ≥ 100%
- `payment_terms` → `refusal` must be ≥ 100%
- `payment_terms` → `language` must be ≥ 100%
- `contract_duration` → `contains` must be ≥ 100%
- `contract_duration` → `refusal` must be ≥ 100%
- `contract_duration` → `language` must be ≥ 100%
- `french_summary` → `contains` must be ≥ 100%
- `french_summary` → `citation` must be ≥ 100%
- `french_summary` → `refusal` must be ≥ 100%
- `french_summary` → `language` must be ≥ 50%
- `french_deadline` → `contains` must be ≥ 100%
- `french_deadline` → `refusal` must be ≥ 100%
- `french_deadline` → `language` must be ≥ 50%
- `arabic_summary` → `contains` must be ≥ 100%
- `arabic_summary` → `citation` must be ≥ 100%
- `arabic_summary` → `refusal` must be ≥ 100%
- `arabic_amount` → `contains` must be ≥ 100%
- `arabic_amount` → `refusal` must be ≥ 100%
- `out_of_scope_ceo` → `contains` must be ≥ 100%
- `out_of_scope_ceo` → `citation` must be ≥ 100%
- `out_of_scope_ceo` → `refusal` must be ≥ 100%
- `out_of_scope_ceo` → `language` must be ≥ 100%
- `out_of_scope_prediction` → `contains` must be ≥ 100%
- `out_of_scope_prediction` → `citation` must be ≥ 100%
- `out_of_scope_prediction` → `refusal` must be ≥ 100%
- `out_of_scope_prediction` → `language` must be ≥ 100%
- `ambiguous_question` → `contains` must be ≥ 100%
- `ambiguous_question` → `citation` must be ≥ 100%
- `ambiguous_question` → `refusal` must be ≥ 100%
- `ambiguous_question` → `language` must be ≥ 100%

## Methodology
1. Retrieve top-5 chunks from Qdrant per question.
2. Feed chunks + question into each prompt variant.
3. Score answers on: content accuracy, hallucination avoidance, citation compliance, refusal correctness, language match.
4. Cache all responses to survive rate-limit interruptions.
5. Average per-case scores to determine the winner.

---
*Generated automatically by `test_prompt_variants.py`*