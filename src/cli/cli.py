import argparse
from pathlib import Path
from ..query_processor.rag_query_processor import RagQueryProcessor
from ..query_processor.fine_tune_query_processor import FineTuneQueryProcessor
from ..data.ingest import load_and_clean_data
from ..rag.index import RagIndex
from ..utils.guardrails import input_guardrails, output_guardrails
from ..rag.memory import MemoryBank
from ..query_processor.prompts import build_prompt, create_context
from ..fine_tune.train import fine_tune_model
from ..test.evaluate_model import benchmark_models
from ..service.chart_boat_service import process_query

def chat_cmd(args):
    print("Starting chat...")
    print("Using model:", args.mode)

    index_dir = Path(args.index_dir)
    memory_path = Path(args.memory_path)
    model_size = args.model_size
    model_path = args.model_path
    mode = args.mode
    
    memory_bank = MemoryBank(memory_path)
    index = None

    if mode == "rag":
       index = RagIndex.load(index_dir)
       query_processor = RagQueryProcessor(model_size=model_size)
    elif mode == "fine-tuning":
       query_processor = FineTuneQueryProcessor(model_path=model_path)

    while True:
        try:
            q = input("\nQ> ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            model_response = process_query(q, query_processor, memory_bank, index)
            if not model_response.get("success", True):
                print("Query processing failed. Please try again.")
                continue
            print(f"A> {model_response.get('answer', 'No answer provided')}")
        except (EOFError, KeyboardInterrupt):
            print() ; break
        
'''
def process_query(query, query_processor, memory_bank, index):
    guardrail_IR = input_guardrails(query)
    if not guardrail_IR.get("status", True):
            print(f"[red]Blocked[/]: {guardrail_IR.get('reason')}")
            return False
    
    ctx = create_context(query, memory_bank, index)
    prompt = build_prompt(ctx, query)

    ans = query_processor.process_query(prompt)

    go = output_guardrails(ans, ctx)
    if not go["ok"]:
        ans += f"\n[Guardrail flags: {', '.join(go['flags'])}]"

    print(f"[bold yellow]Bot[/]: {ans}")

    return True
'''

def ingest_cmd(args):
    print("Ingesting data...")
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    if not raw_dir.exists():
        print(f"Raw directory {raw_dir} does not exist.")
        return
    if not out_dir.exists():
        print(f"Output directory {out_dir} does not exist. Creating it.")
        out_dir.mkdir(parents=True, exist_ok=True)

    load_and_clean_data(raw_dir, out_dir)


def build_cmd(args):
    print("Building index...")
    processed_dir = Path(args.processed_dir)
    index_dir = Path(args.index_dir)
    if not processed_dir.exists():
        print(f"Processed directory {processed_dir} does not exist.")
        return
    if not index_dir.exists():
        print(f"Index directory {index_dir} does not exist. Creating it.")
        index_dir.mkdir(parents=True, exist_ok=True)

    emb_model = args.emb_model

    print(f"Initialized RagIndex with processed_dir: {processed_dir}, index_dir: {index_dir}, emb_model: {emb_model}")
    
    rag_index = RagIndex(emb_model)
    rag_index.build_index(processed_dir)
    rag_index.save(index_dir)

    print(f"Index saved to {index_dir}")

def finetune_cmd(args):
    print("Fine-tuning model...")
    qa_path = Path(args.qa_path)
    output_dir = Path(args.output_dir)
    model_size = args.model_size
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    if not qa_path.exists():
        print(f"Q&A path {qa_path} does not exist.")
        return
    if not output_dir.exists():
        print(f"Output directory {output_dir} does not exist. Creating it.")
        output_dir.mkdir(parents=True, exist_ok=True)

    #print(f"Fine-tuning with model_size: {model_size}, qa_path: {qa_path}, output_dir: {output_dir}, lr: {lr}, epochs: {epochs}, batch_size: {batch_size}")    
    model_out_dir = fine_tune_model(qa_path, output_dir, model_size, lr, epochs, batch_size)
    print(f"Fine-tuned model saved to: {model_out_dir}")

def benchmark_cmd(args):
    print("Benchmarking model...")
    qa_path = Path(args.qa_path)
    index_dir =  Path(args.index_dir)
    model_size = args.model_size
    model_path = args.model_path

    if not qa_path.exists():
        print(f"Q&A path {qa_path} does not exist.")
        return
    if not index_dir.exists():
        print(f"Index directory {index_dir} does not exist.")
        return

    df = benchmark_models(qa_path, model_size=model_size, model_path=model_path, index_dir=index_dir)
    
    output_file = Path(args.output_file)
    output_path = Path(output_file).parent
    if not output_path.exists():
        print(f"Output directory {output_path} does not exist. Creating it.")
        output_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)
    print(f"Benchmark results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Financial QA AI Chartbot")
    sub = parser.add_subparsers(dest="cmd")
    sub.required = True

    p_ingest = sub.add_parser("ingest", help="Ingest + clean + segment + chunk (creates multiple chunk sizes)")
    p_ingest.add_argument("--raw_dir", default="data/raw", help="Path to the data to ingest")
    p_ingest.add_argument("--out_dir", default="data/processed", help="Path to the processed data output directory")
    p_ingest.set_defaults(func=ingest_cmd)


    p_chat = sub.add_parser("chat", help="Interactive chat over built index")
    p_chat.add_argument("--mode", default="rag", choices=["rag", "fine-tuning"], help="LLM engine to use")
    p_chat.add_argument("--index_dir", default="data/index", help="Path to the index directory")
    p_chat.add_argument("--memory_path", default="data/memory/memory_bank.csv", help="Path to the memory directory")
    p_chat.add_argument("--model_size", default="small", choices=["small" , "medium"], help="Size of the model to use")
    p_chat.add_argument("--model_path", default="fine_tune_models/", help="Path to the model directory")
    p_chat.add_argument("--max_ctx_chunks", type=int, default=6, help="Max context chunks to use for RAG")
    p_chat.set_defaults(func=chat_cmd)

    p_build = sub.add_parser("build", help="Build hybrid index (FAISS + BM25)")
    p_build.add_argument("--processed_dir", default="data/processed", help="Path to the processed data directory")
    p_build.add_argument("--index_dir", default="data/index", help="Path to save the built index")
    p_build.add_argument("--emb_model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model to use for indexing")
    p_build.set_defaults(func=build_cmd)    


    p_finetune = sub.add_parser("finetune", help="Fine-tune GPT-2 on your Q&A pairs")
    p_finetune.add_argument("--qa_path", default="data/finetune/microsoft_finetune_qa.csv", help="Path to the question answare pairs csv file for fine-tuning")
    p_finetune.add_argument("--output_dir", default="fine_tune_models/", help="Path to save the fine-tuned model")
    p_finetune.add_argument("--model_size", default="small", choices=["small" , "medium"], help="Size of the model to use")
    p_finetune.add_argument("--lr", default=5e-5, type=float, help="Learning rate for fine-tuning")
    p_finetune.add_argument("--epochs", default=5, type=int, help="Number of epochs for fine-tuning")
    p_finetune.add_argument("--batch_size", default=2, type=int, help="Batch size for fine-tuning")
    p_finetune.set_defaults(func=finetune_cmd)

    p_benchmark = sub.add_parser("benchmark", help="Benchmark base model (pre-fine-tuning) on test questions")
    p_benchmark.add_argument("--qa_path", default="data/finetune/microsoft_finetune_qa.csv", help="Path to the question answer pairs csv file for benchmarking")
    p_benchmark.add_argument("--index_dir", default="data/index", help="Path to the index directory")
    p_benchmark.add_argument("--model_size", default="small", choices=["small" , "medium"], help="Size of the model to use")
    p_benchmark.add_argument("--model_path", default="fine_tune_models/", help="Path to the model directory")
    p_benchmark.add_argument("--output_file", default="model_benchmark/benchmark.csv", help="Path to the benchmark directory")
    p_benchmark.set_defaults(func=benchmark_cmd)

    args = parser.parse_args()
    args.func(args)



         

    

if __name__ == "__main__":
    main()