import sys
import dataclasses
from typing import List, Any
from dotenv import load_dotenv

# Ensure dotenv is loaded first to get API keys and DB secrets
load_dotenv()

from reformulation.llm.config import LLMConfig
from reformulation.llm.client import LLMClient
from reformulation.reformulator import QueryReformulator
from reformulation.semantic_similarity import SemanticSimilarity
from retrieval.vector_retriever import VectorRetriever
from retrieval.keyword_retriever import KeywordRetriever
from retrieval.hybrid_retriever import HybridRetriever
from reformulation.search_pipeline import SearchPipeline
from reformulation.normalizer import normalize_query
from retrieval.models import RetrievedChunk


class InteractiveDemo:
    def __init__(self):
        try:
            llm_config = LLMConfig.from_env()
            llm_client = LLMClient(config=llm_config)
            self.reformulator = QueryReformulator(llm_client=llm_client)
            self.semantic_similarity = SemanticSimilarity()
            vector_retriever = VectorRetriever()
            keyword_retriever = KeywordRetriever()
            self.hybrid_retriever = HybridRetriever(
                vector_retriever=vector_retriever,
                keyword_retriever=keyword_retriever
            )
            
            self.pipeline = SearchPipeline(
                reformulator=self.reformulator,
                semantic_similarity=self.semantic_similarity,
                hybrid_retriever=self.hybrid_retriever
            )
        except Exception as e:
            print(f"Failed to initialize components: {e}")
            sys.exit(1)
        
    def run(self):
        print("============================================================")
        print("QUERY REFORMULATION ENGINE")
        print("============================================================")
        print("\nKnowledge Base")
        print("Documents : 157")
        print("Chunks    : 696\n")

        # Hook methods to log steps without duplicating logic or network requests
        orig_generate = self.reformulator.generate_variants
        orig_filter = self.semantic_similarity.filter_redundant
        orig_retrieve = self.hybrid_retriever.retrieve

        def logged_generate(query, max_variants=5):
            print("\n[2] NORMALIZED QUERY")
            normalized = normalize_query(query)
            print(f"> {normalized}")
            self._current_normalized_query = normalized
            
            print("\n[3] LLM REFORMULATIONS")
            try:
                variants = orig_generate(query, max_variants=max_variants)
                for i, v in enumerate(variants, 1):
                    print(f"  {i}. {v}")
                if not variants:
                    print("  (No variants generated)")
                return variants
            except Exception as e:
                print(f"  [!] LLM error (using original query only): {e}")
                # Re-raise so pipeline falls back safely
                raise

        def logged_filter(queries):
            print("\n[4] SEMANTIC FILTERING")
            try:
                final_qs = orig_filter(queries)
                print(f"  Candidates : {len(queries)}")
                print(f"  Retained   : {len(final_qs)}")
                removed = [q for q in queries if q not in final_qs]
                if removed:
                    print("  Removed queries:")
                    for q in removed:
                        print(f"    - {q}")
                return final_qs
            except Exception as e:
                print(f"  [!] Semantic filtering failed: {e}")
                raise

        def logged_retrieve(*args, **kwargs):
            q = kwargs.get('query') or args[0]
            results = orig_retrieve(*args, **kwargs)
            
            is_original = getattr(self, '_current_normalized_query', None) == q

            if is_original:
                if not getattr(self, '_printed_orig_retrieval', False):
                    print("\n[5] ORIGINAL QUERY RETRIEVAL")
                    self._printed_orig_retrieval = True
                
                print(f"Query: {q}")
                print("Type: ORIGINAL")
                for idx, chunk in enumerate(results[:5], 1):
                    print(f"  Rank: {idx} | Score: {chunk.score:.4f}")
                    print(f"  Title: {chunk.title}")
                    print(f"  Doc ID: {chunk.document_id} | Chunk ID: {chunk.chunk_id}")
                    if chunk.section:
                        print(f"  Section: {chunk.section}")
                    print()
            else:
                if not getattr(self, '_printed_reform_retrieval', False):
                    print("\n[6] REFORMULATED QUERY RETRIEVAL")
                    self._printed_reform_retrieval = True
                    
                print(f"\nQuery: {q}")
                print("Type: REFORMULATED")
                if not results:
                    print("  (No results)")
                for idx, chunk in enumerate(results[:3], 1):
                    print(f"  Rank: {idx} | Score: {chunk.score:.4f} | Doc: {chunk.document_id}")
                    print(f"  Title: {chunk.title}")
            return results

        while True:
            try:
                raw_input = input("\nEnter your query (or type 'exit'):\n> ")
                if not raw_input or not raw_input.strip():
                    continue
                if raw_input.strip().lower() in ('exit', 'quit'):
                    break
                query = raw_input

                print("\n[1] ORIGINAL QUERY")
                print(f"> {query}")

                self._current_normalized_query = None
                self._printed_orig_retrieval = False
                self._printed_reform_retrieval = False
                
                # Apply hooks safely within try/finally
                self.reformulator.generate_variants = logged_generate
                self.semantic_similarity.filter_redundant = logged_filter
                self.hybrid_retriever.retrieve = logged_retrieve
                
                try:
                    final_results = self.pipeline.search(
                        query=query, 
                        max_variants=5, 
                        vector_match_count=10, 
                        keyword_match_count=10, 
                        k=60, 
                        final_k=10
                    )
                finally:
                    # Restore original methods immediately after the pipeline finishes processing this query
                    self.reformulator.generate_variants = orig_generate
                    self.semantic_similarity.filter_redundant = orig_filter
                    self.hybrid_retriever.retrieve = orig_retrieve
                
                print("\n[7] FINAL RRF RESULTS")
                if not final_results:
                    print("No results found.")
                else:
                    for idx, chunk in enumerate(final_results[:5], 1):
                        print(f"  Final Rank: {idx} | Score: {chunk.score:.4f}")
                        print(f"  Title: {chunk.title}")
                        print(f"  Doc ID: {chunk.document_id} | Chunk ID: {chunk.chunk_id}")
                        if chunk.section:
                            print(f"  Section: {chunk.section}")
                        print()
                        
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n[!] An error occurred during search: {e}")
                
if __name__ == "__main__":
    demo = InteractiveDemo()
    demo.run()
