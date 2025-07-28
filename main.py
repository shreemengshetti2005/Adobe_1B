import os
import json
from src.pdf_parser import extract_sections_from_pdf
from src.ranker import rank_sections_by_relevance, rerank_for_diversity
from src.snippet_extractor import extract_top_snippets
from src.output import generate_json


def find_collection_folders():
    """Yield all valid collection paths inside Challenge_1b/."""
    base_dir = "Challenge_1b"
    if os.path.exists(base_dir):
        for name in sorted(os.listdir(base_dir)):
            collection_path = os.path.join(base_dir, name)
            if os.path.isdir(collection_path) and "challenge1b_input.json" in os.listdir(collection_path):
                yield collection_path


def process_collection(collection_path):
    if collection_path:
        print(f"Processing collection: {collection_path}")
        input_json_path = os.path.join(collection_path, "challenge1b_input.json")
        output_json_path = os.path.join(collection_path, "challenge1b_output.json")
        pdf_folder = os.path.join(collection_path, "PDFs")
    else:
        print("No structured collection found. Using fallback: ./input and ./output")
        input_json_path = os.path.join("input", "challenge1b_input.json")
        output_json_path = os.path.join("output", "challenge1b_output.json")
        pdf_folder = "input"
        os.makedirs("output", exist_ok=True)

    # 1. Load input JSON
    if not os.path.exists(input_json_path):
        print(f"Error: Input file not found: {input_json_path}")
        return

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    persona = data.get("persona", {}).get("role", "").strip()
    job = data.get("job_to_be_done", {}).get("task", "").strip()
    docs_info = data.get("documents", [])
    doc_filenames = [d.get("filename") for d in docs_info]

    # 2. Extract sections from PDFs
    all_sections = []
    print("Extracting sections from PDFs...")
    for pdf_filename in doc_filenames:
        pdf_path = os.path.join(pdf_folder, pdf_filename)
        if os.path.exists(pdf_path):
            sections = extract_sections_from_pdf(pdf_path)
            all_sections.extend(sections)
            print(f"Extracted {len(sections)} sections from {pdf_filename}")
        else:
            print(f"Warning: Skipping missing file {pdf_filename}")

    if not all_sections:
        print("Error: No sections were extracted from any PDF.")
        return

    # 3. Rank all sections
    print("Ranking sections for relevance...")
    ranked_sections = rank_sections_by_relevance(all_sections, persona, job)
    if not ranked_sections:
        print("Error: Section ranking failed.")
        return

    # 4. Re-rank for diversity
    print("Applying diversity re-ranking...")
    diverse_top_sections = rerank_for_diversity(ranked_sections, top_n=5)

    print("Top 5 selected sections:")
    for i, sec in enumerate(diverse_top_sections, 1):
        print(
            f"{i}. {sec['document']} (page {sec['page_number']}): {sec['section_title']} | Score: {sec['relevance_score']:.3f}")

    # 5. Extract snippets
    print("Extracting snippets...")
    final_sections_with_snippets = []
    for idx, sec in enumerate(diverse_top_sections, 1):
        snippets = extract_top_snippets(
            section_content=sec["content"],
            header=sec["section_title"],
            persona=persona,
            job_to_be_done=job,
            top_k=5
        )
        if not snippets:
            first_sentence = sec.get("content", "").split('.')[0]
            snippets = [{"refined_text": first_sentence + '.', "relevance_score": 0.0}]

        sec["top_snippets"] = snippets
        sec["importance_rank"] = idx
        final_sections_with_snippets.append(sec)

    # 6. Generate and write output
    print("Generating output JSON...")
    final_output_str = generate_json(
        input_documents=[d.get("title", "") for d in docs_info],
        persona=persona,
        job_to_be_done=job,
        ranked_sections_with_snippets=final_sections_with_snippets
    )

    with open(output_json_path, "w", encoding="utf-8") as f:
        f.write(final_output_str)

    print(f"Output written to {output_json_path}")


def main():
    any_processed = False
    for collection_path in find_collection_folders():
        process_collection(collection_path)
        any_processed = True

    if not any_processed:
        process_collection(None)


if __name__ == "__main__":
    main()
