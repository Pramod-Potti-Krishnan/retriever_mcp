-- Database Schema for Document Retrieval MCP Server
-- Run this in your Supabase SQL editor

-- Enable pgvector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL,
    session_id text NOT NULL,
    project_id text DEFAULT '-',
    filename text NOT NULL,
    file_type text NOT NULL,
    file_size integer,
    total_chunks integer,
    upload_date timestamp with time zone DEFAULT now(),
    processing_status text DEFAULT 'completed',
    metadata jsonb,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);

-- Create document_embeddings table
CREATE TABLE IF NOT EXISTS document_embeddings (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid REFERENCES documents(id) ON DELETE CASCADE,
    user_id uuid NOT NULL,
    session_id text NOT NULL,
    project_id text DEFAULT '-',
    chunk_text text NOT NULL,
    embedding vector(1536), -- For OpenAI text-embedding-3-small
    chunk_index integer NOT NULL,
    chunk_metadata jsonb,
    created_at timestamp with time zone DEFAULT now()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_embeddings_vector
ON document_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_embeddings_user_session
ON document_embeddings(user_id, session_id);

CREATE INDEX IF NOT EXISTS idx_documents_user_session
ON documents(user_id, session_id);

CREATE INDEX IF NOT EXISTS idx_embeddings_document
ON document_embeddings(document_id);

CREATE INDEX IF NOT EXISTS idx_embeddings_project
ON document_embeddings(project_id);

-- Create the search_embeddings function (optional, for RPC-based search)
CREATE OR REPLACE FUNCTION search_embeddings(
    query_embedding vector,
    user_id uuid,
    session_id text,
    project_id text DEFAULT '-',
    similarity_threshold float DEFAULT 0.7,
    match_count int DEFAULT 5
)
RETURNS TABLE (
    id uuid,
    document_id uuid,
    chunk_text text,
    chunk_index int,
    chunk_metadata jsonb,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        de.id,
        de.document_id,
        de.chunk_text,
        de.chunk_index,
        de.chunk_metadata,
        1 - (de.embedding <=> query_embedding) as similarity
    FROM document_embeddings de
    WHERE
        de.user_id = search_embeddings.user_id
        AND de.session_id = search_embeddings.session_id
        AND de.project_id = search_embeddings.project_id
        AND 1 - (de.embedding <=> query_embedding) > similarity_threshold
    ORDER BY de.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Grant necessary permissions (adjust as needed)
GRANT ALL ON documents TO authenticated;
GRANT ALL ON document_embeddings TO authenticated;
GRANT EXECUTE ON FUNCTION search_embeddings TO authenticated;

-- Optional: Add RLS policies if needed
-- ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE document_embeddings ENABLE ROW LEVEL SECURITY;