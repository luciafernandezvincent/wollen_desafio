import os, re
from typing import List, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# State 
class ResearchState(TypedDict):
    topic: str
    subtopics: List[str]
    approved_topics: List[str]
    sources: List[str]
    analysis_data: str
    final_report: str
    next_step: str

#  Model + Mock Tool 
def get_model(tier="cheap"):
    return ChatOpenAI(
        model="Mistral-Small-2503",
        api_key=os.environ.get("MISTRAL_API_KEY"),
        base_url="https://models.inference.ai.azure.com",
        temperature=0.1 if tier == "cheap" else 0.7
    )

def mock_search_tool(topic: str):
    """
    Enhanced Mock Search: Generates diverse 'data' perspectives 
    for ANY topic to ground the LLM's research.
    """
    return [
        {
            "url": f"https://tech-pedia.org/wiki/{topic.replace(' ', '_')}", 
            "content": f"The technical architecture of {topic} involves complex systems integration and modern frameworks. Recent benchmarks show a 20% efficiency increase in the latest version."
        },
        {
            "url": "https://industry-observer.com/reports/global-trends", 
            "content": f"Market analysis indicates that {topic} is seeing a surge in adoption across North America and Europe. Key drivers include cost reduction and scalability requirements."
        },
        {
            "url": "https://future-science.edu/publications", 
            "content": f"A recent study on {topic} highlights several ethical considerations and sustainability challenges. Researchers suggest that decentralized approaches may mitigate current risks."
        }
    ]
#  Nodes
def supervisor_node(state: ResearchState):
    print("\n[Supervisor] Evaluation...")
    if not state.get("subtopics"): return {"next_step": "investigator"}
    # If approved_topics is empty,  stay in human_pause
    if not state.get("approved_topics") or len(state.get("approved_topics")) == 0:
        return {"next_step": "human_pause"}
    if not state.get("analysis_data"): return {"next_step": "curator"}
    if not state.get("final_report"): return {"next_step": "reporter"}
    return {"next_step": "end"}

def investigator_node(state: ResearchState):
    print(f"\n[Investigator] Researching {state['topic']}...")
    llm = get_model("cheap")
    res = llm.invoke(f"List 5 specific sub-topics for {state['topic']}. Bullet points only.")
    topics = [l.strip("- ").strip("12345. ") for l in res.content.split("\n") if l.strip()]
    return {"subtopics": topics, "sources": ["https://mock-source.com"]}

def curator_node(state: ResearchState):
    print(f"\n[Curator] Analyzing: {state['approved_topics']}...")
    llm = get_model("cheap")
    res = llm.invoke(f"Write a technical analysis for: {state['approved_topics']}")
    return {"analysis_data": res.content}

def reporter_node(state: ResearchState):
    print("\n[Reporter] Finalizing report...")
    llm = get_model("expensive")
    res = llm.invoke(f"Markdown report from: {state['analysis_data']}")
    return {"final_report": res.content}

# Fixed Command Parser 
def human_validation_step(subtopics: List[str]) -> List[str]:
    print("\n" "SUPERVISOR GATE")
    current = list(subtopics)
    
    while True:
        print("\nPROPOSED TOPICS:")
        for i, t in enumerate(current): print(f"  [{i+1}] {t}")
        print("\nCOMMANDS: 'approve 1,3' | 'reject 2' | 'modify 1 to Name' | 'ok'")
        
        user_input = input("Decision: ").strip().lower()
        if user_input == 'ok': return current # Returns whatever is currently in the list

        try:
            if user_input.startswith("approve"):
                indices = [int(n.strip()) - 1 for n in user_input.replace("approve", "").split(",")]
                current = [current[i] for i in indices if 0 <= i < len(current)]
                print("Selection kept.")
            elif user_input.startswith("reject"):
                idx = int(user_input.split()[1]) - 1
                current.pop(idx)
                print("Removed.")
            elif user_input.startswith("modify"):
                match = re.search(r"modify (\d+) to (.+)", user_input)
                idx = int(match.group(1)) - 1
                current[idx] = match.group(2).strip()
                print(f"Modified item {idx+1}")
            else:
                print("Unknown command.")
        except:
            print("Formatting error.")

# Graph 
workflow = StateGraph(ResearchState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("investigator", investigator_node)
workflow.add_node("curator", curator_node)
workflow.add_node("reporter", reporter_node)

workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", lambda x: x["next_step"], 
    {"investigator": "investigator", "curator": "curator", "reporter": "reporter", "human_pause": END, "end": END})
workflow.add_edge("investigator", "supervisor")
workflow.add_edge("curator", "supervisor")
workflow.add_edge("reporter", "supervisor")

app = workflow.compile(checkpointer=MemorySaver())


def main():
    config = {"configurable": {"thread_id": "interview_v3"}}
    topic = input("Enter Topic: ")
    
    app.invoke({"topic": topic}, config)
    
    snapshot = app.get_state(config)
    ai_found = snapshot.values.get('subtopics', [])
    user_final_list = human_validation_step(ai_found)
    
   
    print("\n[System] Applying edits and resuming...")
    app.update_state(config, 
                    {"approved_topics": user_final_list, "next_step": "curator"}, 
                    as_node="supervisor")
    
    final_output = None
    for event in app.stream(None, config, stream_mode="values"):
        final_output = event
        
    if final_output and "final_report" in final_output:
        report_content = final_output["final_report"]
        
        print("\n" + "="*40 + "\nFINAL REPORT GENERATED\n" + "="*40)
        print(report_content)

        # save to file
        safe_name = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_').lower()
        filename = f"research_{safe_name}.md"
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_content)
            print(f"\n Report saved as: {os.getcwd()}/{filename}")
        except Exception as e:
            print(f"Failed to save file: {e}")
            
    else:
        print(f"Error: Ended at {final_output.get('next_step')}")

if __name__ == "__main__":
    main()