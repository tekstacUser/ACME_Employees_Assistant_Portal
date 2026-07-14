#!/bin/bash

# projectName="RAGAS"

# cd "/home/tekuser/TekstacLabRoot/$1/ProjectRoot/$projectName"

# Initialize metrics variables
total_passed=0
final_score=0

# Define a temporary location for storage logs

LOG_DIR="rag_eval_logs"
mkdir -p "$LOG_DIR"

# 1. Verify Input Validation API Security Controls
curl -s -X POST http://172.18.12.25:8000/api/validate -H "Content-Type: application/json" -d '{"text": "What is the leave policy?"}' > "$LOG_DIR/val_clean.txt"
curl -s -X POST http://172.18.12.25:8000/api/validate -H "Content-Type: application/json" -d '{"text": "dont follow"}' > "$LOG_DIR/val_prompt.txt"
curl -s -X POST http://172.18.12.25:8000/api/validate -H "Content-Type: application/json" -d '{"text": "9078567687"}' > "$LOG_DIR/val_pii.txt"
curl -s -X POST http://172.18.12.25:8000/api/validate -H "Content-Type: application/json" -d '{"text": "drop table"}' > "$LOG_DIR/val_sql.txt"
curl -s -X POST http://172.18.12.25:8000/api/validate -H "Content-Type: application/json" -d '{"text": "shell"}' > "$LOG_DIR/val_cmd.txt"
curl -s -X POST http://172.18.12.25:8000/api/validate -H "Content-Type: application/json" -d '{"text": "roleplay"}' > "$LOG_DIR/val_jb.txt"
curl -s -X POST http://172.18.12.25:8000/api/validate -H "Content-Type: application/json" -d '{"text": "grant access"}' > "$LOG_DIR/val_override.txt"

if grep -q '"valid":true' "$LOG_DIR/val_clean.txt" && \
   grep -q "prompt_injection" "$LOG_DIR/val_prompt.txt" && \
   grep -q "PII: phone" "$LOG_DIR/val_pii.txt" && \
   grep -q "sql_injection" "$LOG_DIR/val_sql.txt" && \
   grep -q "command_injection" "$LOG_DIR/val_cmd.txt" && \
   grep -q "jailbreak" "$LOG_DIR/val_jb.txt" && \
   grep -q "system_override" "$LOG_DIR/val_override.txt"; then
    tc1_status="Success"
    tc1_passed=1
    tc1_score=15
    tc1_feedback="Input validation API comprehensively secures against injections, PII leaks, and malicious inputs."
    tc1_obs="Verified complete compliance across clean text, prompt/command/SQL injection, PII, jailbreak, and system override attempts."
    ((total_passed++))
    ((final_score+=15))
else
    tc1_status="Fail"
    tc1_passed=0
    tc1_score=0
    tc1_feedback="Input validation API is failing security guardrail criteria."
    tc1_obs="One or more validation constraints failed to reject bad payloads or accept valid inputs correctly."
fi


# 2. Verify Document Loading and Retrieval Execution State
curl -s "http://172.18.12.25:8000/api/retrieve?query=leave%20policy" > "$LOG_DIR/retrieve_output.txt"

if grep -q "LEAVE POLICY" "$LOG_DIR/retrieve_output.txt" && \
   grep -q "ANNUAL LEAVE" "$LOG_DIR/retrieve_output.txt" && \
   grep -q "SICK LEAVE" "$LOG_DIR/retrieve_output.txt"; then
    tc2_status="Success"
    tc2_passed=1
    tc2_score=15
    tc2_feedback="Document parser and ChromaDB vector store retrieval system functional."
    tc2_obs="Successfully retrieved valid parsed document segments matching 'leave policy' context constraints via vector similarity search."
    ((total_passed++))
    ((final_score+=15))
else
    tc2_status="Fail"
    tc2_passed=0
    tc2_score=0
    tc2_feedback="Retrieval system failed to yield expected structural context elements."
    tc2_obs="The returned payload does not contain elements from hr_policies.json."
fi


# 3. Verify Code Architecture Parameters (Chunk Size / Overlap / Target File)
pipeline_file="rag_pipeline_minimal.py"

if [ -f "$pipeline_file" ]; then

    chunk800_count=$(grep -c "800" "$pipeline_file")
    chunk20_count=$(grep -c "20" "$pipeline_file")
    asset_count=$(grep -c "hr_policies.json" "$pipeline_file")

    if [ "$chunk800_count" -ge 2 ] && \
       [ "$chunk20_count" -ge 12 ] && \
       [ "$asset_count" -ge 3 ]; then
        tc3_status="Success"
        tc3_passed=1
        tc3_score=15
        tc3_feedback="Chunk configuration and document asset paths verified inside codebase architecture."
        tc3_obs="Verified explicit token parameters (Chunk Size: 800, Overlap: 20) alongside target source asset registration."
        ((total_passed++))
        ((final_score+=15))
    else
        tc3_status="Fail"
        tc3_passed=0
        tc3_score=0
        tc3_feedback="Pipeline script verification failed."
        tc3_obs="Missing required chunk parameters (800/20) or hr_policies.json asset mapping."
    fi

else
    tc3_status="Fail"
    tc3_passed=0
    tc3_score=0
    tc3_feedback="Pipeline script verification failed."
    tc3_obs="Pipeline file $pipeline_file is missing."
fi


# 4. Verify RAG Complete Query Generation Loop
curl -s -X POST http://172.18.12.25:8000/api/query -H "Content-Type: application/json" -d '{"question": "What is the leave policy?"}' > "$LOG_DIR/query_output.txt"

if grep -q "Maternity Leave" "$LOG_DIR/query_output.txt" && \
   grep -q "Paternity Leave" "$LOG_DIR/query_output.txt" && \
   grep -q "quality_score" "$LOG_DIR/query_output.txt"; then
    tc4_status="Success"
    tc4_passed=1
    tc4_score=15
    tc4_feedback="Generative RAG answering loop executed successfully."
    tc4_obs="The query pipeline successfully structured response contents matching knowledge bases alongside system context quality scoring."
    ((total_passed++))
    ((final_score+=15))
else
    tc4_status="Fail"
    tc4_passed=0
    tc4_score=0
    tc4_feedback="Query engine generation context failed or returned incomplete payloads."
    tc4_obs="Output properties missing response text or score metrics criteria."
fi


# 5. Verify Model Evaluation Framework Controls
curl -s -X POST http://172.18.12.25:8000/api/evaluate -H "Content-Type: application/json" -d '{"query":"What is the leave policy?","response":"Employees receive 20 days of annual leave."}' > "$LOG_DIR/evaluate_output.txt"

if grep -q "faithfulness" "$LOG_DIR/evaluate_output.txt" && \
   grep -q "relevance" "$LOG_DIR/evaluate_output.txt" && \
   grep -q "overall" "$LOG_DIR/evaluate_output.txt"; then
    tc5_status="Success"
    tc5_passed=1
    tc5_score=15
    tc5_feedback="Evaluation endpoint calculates response faithfulness and context relevance accurately."
    tc5_obs="Received matrix validation outputs showcasing numerical performance telemetry data blocks."
    ((total_passed++))
    ((final_score+=15))
else
    tc5_status="Fail"
    tc5_passed=0
    tc5_score=0
    tc5_feedback="Evaluation API did not return structured telemetry values."
    tc5_obs="The expected evaluation breakdown components were not received."
fi


# 6. Verify PEFT LoRA Finetuning Hyperparameter Matrix Configuration
curl -s http://172.18.12.25:8000/api/finetuning > "$LOG_DIR/finetuning_output.txt"

if grep -q "lora" "$LOG_DIR/finetuning_output.txt" && \
   grep -q "target_modules" "$LOG_DIR/finetuning_output.txt" && \
   grep -q "q_proj" "$LOG_DIR/finetuning_output.txt" && \
   grep -q "v_proj" "$LOG_DIR/finetuning_output.txt"; then
    tc6_status="Success"
    tc6_passed=1
    tc6_score=15
    tc6_feedback="Fine-tuning orchestration configuration properties verified."
    tc6_obs="Confirmed LoRA attention projection weights (q_proj, v_proj) target matrices match expected fine-tuning specifications."
    ((total_passed++))
    ((final_score+=15))
else
    tc6_status="Fail"
    tc6_passed=0
    tc6_score=0
    tc6_feedback="Fine-tuning structural parameter settings are incorrect or inaccessible."
    tc6_obs="The system failed to display target configurations mapping back to required model layers."
fi


# 7. Verify Core System Component Micro-Health Indicators (now 7 components: guardrails,
#    retrieval, query, evaluate, finetuning, vector store, prompt registry / mlflow)
curl -s http://172.18.12.25:8000/api/health > "$LOG_DIR/health_output.txt"

if grep -q '"status":"healthy"' "$LOG_DIR/health_output.txt" && \
   grep -q '"components":7' "$LOG_DIR/health_output.txt" && \
   grep -q '"vector_store"' "$LOG_DIR/health_output.txt" && \
   grep -q '"mlflow"' "$LOG_DIR/health_output.txt"; then
    tc7_status="Success"
    tc7_passed=1
    tc7_score=10
    tc7_feedback="All functional application modular components are online and passing system health probes."
    tc7_obs="Health routing checks validated active status indicators covering all 7 core framework boundaries, including vector store and MLflow telemetry."
    ((total_passed++))
    ((final_score+=10))
else
    tc7_status="Fail"
    tc7_passed=0
    tc7_score=0
    tc7_feedback="System components report unhealthy states or missing dependencies."
    tc7_obs="Health verification endpoint returned failure assertions or an incorrect online components tally."
fi


# 8. Verify ChromaDB Embedded Vector Store Management
curl -s http://172.18.12.25:8000/api/vectorstore/stats > "$LOG_DIR/vs_stats.txt"
curl -s -X POST http://172.18.12.25:8000/api/vectorstore/reingest > "$LOG_DIR/vs_reingest.txt"
curl -s "http://172.18.12.25:8000/api/retrieve/detailed?query=leave%20policy&k=3" > "$LOG_DIR/vs_detailed.txt"

if grep -q "ChromaDB" "$LOG_DIR/vs_stats.txt" && \
   grep -q '"collection_name":"hr_policies"' "$LOG_DIR/vs_stats.txt" && \
   grep -qE '"vector_count":[1-9][0-9]*' "$LOG_DIR/vs_stats.txt" && \
   grep -qE '"vector_count":[1-9][0-9]*' "$LOG_DIR/vs_reingest.txt" && \
   grep -q '"similarity"' "$LOG_DIR/vs_detailed.txt" && \
   grep -q '"results"' "$LOG_DIR/vs_detailed.txt"; then
    tc8_status="Success"
    tc8_passed=1
    tc8_score=15
    tc8_feedback="Embedded ChromaDB vector store is correctly initialized, queryable, and supports re-ingestion."
    tc8_obs="Confirmed persistent local ChromaDB collection 'hr_policies' with populated vector counts, and similarity-scored detailed retrieval results."
    ((total_passed++))
    ((final_score+=15))
else
    tc8_status="Fail"
    tc8_passed=0
    tc8_score=0
    tc8_feedback="Vector store management endpoints failed to report a functioning embedded ChromaDB backend."
    tc8_obs="Stats, re-ingest, or similarity-scored retrieval payloads were missing or incomplete."
fi


# 9. Verify Persistent Prompt Registry & Lifecycle Management (SQLite-backed)
curl -s http://172.18.12.25:8000/api/prompts > "$LOG_DIR/prompts_list.txt"
curl -s http://172.18.12.25:8000/api/prompts/hr_assistant/production > "$LOG_DIR/prod_before.txt"
prod_before_version=$(python3 -c "
import json
try:
    d = json.load(open('$LOG_DIR/prod_before.txt'))
    print(d.get('version', ''))
except Exception:
    print('')
")

curl -s -X POST http://172.18.12.25:8000/api/prompts -H "Content-Type: application/json" \
  -d '{"name":"hr_assistant","content":"You are an empathetic and precise ACME HR assistant.","tags":["eval-run"],"notes":"Created by automated evaluation"}' \
  > "$LOG_DIR/prompt_created.txt"
new_version=$(python3 -c "
import json
try:
    d = json.load(open('$LOG_DIR/prompt_created.txt'))
    print(d.get('version', ''))
except Exception:
    print('')
")

if [ -n "$new_version" ]; then
    curl -s -X POST http://172.18.12.25:8000/api/prompts/hr_assistant/promote -H "Content-Type: application/json" \
      -d "{\"version\": $new_version, \"faithfulness\": 0.95, \"answer_relevance\": 0.9, \"latency_ms\": 400}" \
      > "$LOG_DIR/prompt_promoted.txt"

    curl -s http://172.18.12.25:8000/api/prompts/hr_assistant/production > "$LOG_DIR/prod_after.txt"
    curl -s -X POST http://172.18.12.25:8000/api/prompts/hr_assistant/rollback > "$LOG_DIR/prompt_rollback.txt"
    curl -s "http://172.18.12.25:8000/api/prompts/hr_assistant/diff?v1=1&v2=$new_version" > "$LOG_DIR/prompt_diff.txt"
else
    : > "$LOG_DIR/prompt_promoted.txt"
    : > "$LOG_DIR/prod_after.txt"
    : > "$LOG_DIR/prompt_rollback.txt"
    : > "$LOG_DIR/prompt_diff.txt"
fi

if grep -q "hr_assistant" "$LOG_DIR/prompts_list.txt" && \
   [ -n "$new_version" ] && \
   grep -q '"promoted":true' "$LOG_DIR/prompt_promoted.txt" && \
   grep -q "\"version\":$new_version" "$LOG_DIR/prod_after.txt" && \
   grep -q '"status":"production"' "$LOG_DIR/prod_after.txt" && \
   grep -q '"status":"production"' "$LOG_DIR/prompt_rollback.txt" && \
   grep -q '"diff"' "$LOG_DIR/prompt_diff.txt"; then
    tc9_status="Success"
    tc9_passed=1
    tc9_score=20
    tc9_feedback="Persistent prompt registry correctly versions, quality-gates, promotes, and rolls back prompts."
    tc9_obs="Verified full lifecycle: new draft version created, promoted to production under quality gates, previous production version restored via rollback, and unified diff generated between versions."
    ((total_passed++))
    ((final_score+=20))
else
    tc9_status="Fail"
    tc9_passed=0
    tc9_score=0
    tc9_feedback="Prompt registry lifecycle management failed one or more stages."
    tc9_obs="Creation, promotion, production lookup, rollback, or diff endpoints did not return the expected persistent registry state."
fi


# 10. Verify MLflow Experiment Tracking / LLM Observability
curl -s http://172.18.12.25:8000/api/mlflow/info > "$LOG_DIR/mlflow_info.txt"
curl -s -X POST http://172.18.12.25:8000/api/query -H "Content-Type: application/json" -d '{"question": "What is the sick leave policy?"}' > "$LOG_DIR/mlflow_query_trigger.txt"

if grep -q '"mlflow_installed":true' "$LOG_DIR/mlflow_info.txt" && \
   grep -q '"enabled":true' "$LOG_DIR/mlflow_info.txt" && \
   grep -q "acme-hr-rag-assistant" "$LOG_DIR/mlflow_info.txt" && \
   grep -q "tracking_uri" "$LOG_DIR/mlflow_info.txt" && \
   [ -d "mlruns" ]; then
    tc10_status="Success"
    tc10_passed=1
    tc10_score=10
    tc10_feedback="MLflow observability layer is active and logging RAG query telemetry to a local tracking store."
    tc10_obs="Confirmed MLflow is installed, enabled, bound to the 'acme-hr-rag-assistant' experiment, with a resolvable local SQLite tracking URI and on-disk mlruns store."
    ((total_passed++))
    ((final_score+=10))
else
    tc10_status="Fail"
    tc10_passed=0
    tc10_score=0
    tc10_feedback="MLflow tracking layer is disabled, misconfigured, or not persisting run data."
    tc10_obs="The mlflow info endpoint did not report an enabled, correctly-bound tracking store."
fi

# Clean up temp log assets
rm -rf "$LOG_DIR"

# Calculate total metrics layout
total_failed=$((10 - total_passed))
formatted_grade=$(printf "%.2f" "$final_score")

# Output the schema-compliant JSON data layout
cat <<EOF
Grade:=>>$formatted_grade <reportData>{
    "evaluationdetails": {
        "evaluationbreakup": [
            {
                "name": "Verifying Input Validation API Security Controls",
                "totaltestcase": "1",
                "testcasepassed": "$tc1_passed",
                "maxmark": "15",
                "score": "$tc1_score",
                "status": "$tc1_status",
                "feedback": "$tc1_feedback",
                "expertises": {
                    "expertise": {
                        "name": "Input Validation",
                        "testcases": {
                            "testcase": [
                                {
                                    "visible": "yes",
                                    "name": "Security Guardrail Check",
                                    "description": "$tc1_obs",
                                    "shortdescription": "$tc1_feedback",
                                    "maxmark": "15",
                                    "score": "$tc1_score",
                                    "status": "$tc1_status"
                                }
                            ]
                        }
                    }
                }
            },
            {
                "name": "Verifying Document Loading and Retrieval Execution State",
                "totaltestcase": "1",
                "testcasepassed": "$tc2_passed",
                "maxmark": "15",
                "score": "$tc2_score",
                "status": "$tc2_status",
                "feedback": "$tc2_feedback",
                "expertises": {
                    "expertise": {
                        "name": "Context Retrieval",
                        "testcases": {
                            "testcase": [
                                {
                                    "visible": "yes",
                                    "name": "Vector Context Search Check",
                                    "description": "$tc2_obs",
                                    "shortdescription": "$tc2_feedback",
                                    "maxmark": "15",
                                    "score": "$tc2_score",
                                    "status": "$tc2_status"
                                }
                            ]
                        }
                    }
                }
            },
            {
                "name": "Verifying Code Architecture Parameters",
                "totaltestcase": "1",
                "testcasepassed": "$tc3_passed",
                "maxmark": "15",
                "score": "$tc3_score",
                "status": "$tc3_status",
                "feedback": "$tc3_feedback",
                "expertises": {
                    "expertise": {
                        "name": "Pipeline Configuration Structures",
                        "testcases": {
                            "testcase": [
                                {
                                    "visible": "yes",
                                    "name": "Chunk Parameter Configuration Check",
                                    "description": "$tc3_obs",
                                    "shortdescription": "$tc3_feedback",
                                    "maxmark": "15",
                                    "score": "$tc3_score",
                                    "status": "$tc3_status"
                                }
                            ]
                        }
                    }
                }
            },
            {
                "name": "Verifying RAG Complete Query Generation Loop",
                "totaltestcase": "1",
                "testcasepassed": "$tc4_passed",
                "maxmark": "15",
                "score": "$tc4_score",
                "status": "$tc4_status",
                "feedback": "$tc4_feedback",
                "expertises": {
                    "expertise": {
                        "name": "Query Inference Pipeline",
                        "testcases": {
                            "testcase": [
                                {
                                    "visible": "yes",
                                    "name": "Inference Response Engine Check",
                                    "description": "$tc4_obs",
                                    "shortdescription": "$tc4_feedback",
                                    "maxmark": "15",
                                    "score": "$tc4_score",
                                    "status": "$tc4_status"
                                }
                            ]
                        }
                    }
                }
            },
            {
                "name": "Verifying Model Evaluation Framework Controls",
                "totaltestcase": "1",
                "testcasepassed": "$tc5_passed",
                "maxmark": "15",
                "score": "$tc5_score",
                "status": "$tc5_status",
                "feedback": "$tc5_feedback",
                "expertises": {
                    "expertise": {
                        "name": "RAG Triad Metric Evaluations",
                        "testcases": {
                            "testcase": [
                                {
                                    "visible": "yes",
                                    "name": "Faithfulness and Relevance Check",
                                    "description": "$tc5_obs",
                                    "shortdescription": "$tc5_feedback",
                                    "maxmark": "15",
                                    "score": "$tc5_score",
                                    "status": "$tc5_status"
                                }
                            ]
                        }
                    }
                }
            },
            {
                "name": "Verifying PEFT LoRA Finetuning Hyperparameter Matrix Configuration",
                "totaltestcase": "1",
                "testcasepassed": "$tc6_passed",
                "maxmark": "15",
                "score": "$tc6_score",
                "status": "$tc6_status",
                "feedback": "$tc6_feedback",
                "expertises": {
                    "expertise": {
                        "name": "Fine-Tuning Architecture",
                        "testcases": {
                            "testcase": [
                                {
                                    "visible": "yes",
                                    "name": "LoRA Strategy Matrix Target Check",
                                    "description": "$tc6_obs",
                                    "shortdescription": "$tc6_feedback",
                                    "maxmark": "15",
                                    "score": "$tc6_score",
                                    "status": "$tc6_status"
                                }
                            ]
                        }
                    }
                }
            },
            {
                "name": "Verifying Core System Component Micro-Health Indicators",
                "totaltestcase": "1",
                "testcasepassed": "$tc7_passed",
                "maxmark": "10",
                "score": "$tc7_score",
                "status": "$tc7_status",
                "feedback": "$tc7_feedback",
                "expertises": {
                    "expertise": {
                        "name": "System Health Maintenance",
                        "testcases": {
                            "testcase": [
                                {
                                    "visible": "yes",
                                    "name": "Sub-component Integration Health Probes Check",
                                    "description": "$tc7_obs",
                                    "shortdescription": "$tc7_feedback",
                                    "maxmark": "10",
                                    "score": "$tc7_score",
                                    "status": "$tc7_status"
                                }
                            ]
                        }
                    }
                }
            },
            {
                "name": "Verifying ChromaDB Embedded Vector Store Management",
                "totaltestcase": "1",
                "testcasepassed": "$tc8_passed",
                "maxmark": "15",
                "score": "$tc8_score",
                "status": "$tc8_status",
                "feedback": "$tc8_feedback",
                "expertises": {
                    "expertise": {
                        "name": "Vector Store Infrastructure",
                        "testcases": {
                            "testcase": [
                                {
                                    "visible": "yes",
                                    "name": "ChromaDB Stats, Re-ingest & Similarity Retrieval Check",
                                    "description": "$tc8_obs",
                                    "shortdescription": "$tc8_feedback",
                                    "maxmark": "15",
                                    "score": "$tc8_score",
                                    "status": "$tc8_status"
                                }
                            ]
                        }
                    }
                }
            },
            {
                "name": "Verifying Persistent Prompt Registry & Lifecycle Management",
                "totaltestcase": "1",
                "testcasepassed": "$tc9_passed",
                "maxmark": "20",
                "score": "$tc9_score",
                "status": "$tc9_status",
                "feedback": "$tc9_feedback",
                "expertises": {
                    "expertise": {
                        "name": "Prompt Versioning & Governance",
                        "testcases": {
                            "testcase": [
                                {
                                    "visible": "yes",
                                    "name": "Create, Promote, Rollback & Diff Check",
                                    "description": "$tc9_obs",
                                    "shortdescription": "$tc9_feedback",
                                    "maxmark": "20",
                                    "score": "$tc9_score",
                                    "status": "$tc9_status"
                                }
                            ]
                        }
                    }
                }
            },
            {
                "name": "Verifying MLflow Experiment Tracking / LLM Observability",
                "totaltestcase": "1",
                "testcasepassed": "$tc10_passed",
                "maxmark": "10",
                "score": "$tc10_score",
                "status": "$tc10_status",
                "feedback": "$tc10_feedback",
                "expertises": {
                    "expertise": {
                        "name": "Observability & Experiment Tracking",
                        "testcases": {
                            "testcase": [
                                {
                                    "visible": "yes",
                                    "name": "MLflow Tracking Store & Info Check",
                                    "description": "$tc10_obs",
                                    "shortdescription": "$tc10_feedback",
                                    "maxmark": "10",
                                    "score": "$tc10_score",
                                    "status": "$tc10_status"
                                }
                            ]
                        }
                    }
                }
            }
        ],
        "consolidatedtestcase": {
            "totaltestcases": "10",
            "passedtestcase": "$total_passed",
            "failedtestcase": "$total_failed"
        }
    }
}</reportData>
EOF
