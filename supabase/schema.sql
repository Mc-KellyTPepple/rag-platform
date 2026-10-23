-- Run once in the Supabase SQL editor.
create extension if not exists vector;

create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  namespace text not null,
  filename text not null,
  mime text,
  n_chunks int not null default 0,
  created_at timestamptz not null default now()
);

create table if not exists chunks (
  id bigserial primary key,
  document_id uuid not null references documents(id) on delete cascade,
  namespace text not null,
  ord int not null,
  page int,
  section text,
  content text not null,
  embedding vector(384) not null,
  fts tsvector generated always as (to_tsvector('english', content)) stored
);

create index if not exists chunks_embedding_idx on chunks using hnsw (embedding vector_cosine_ops);
create index if not exists chunks_fts_idx on chunks using gin (fts);
create index if not exists chunks_ns_doc_idx on chunks (namespace, document_id);
create index if not exists documents_ns_idx on documents (namespace);

-- Block Supabase's public REST API from reading these tables.
-- The backend connects with the database role directly, which bypasses RLS.
alter table documents enable row level security;
alter table chunks enable row level security;

-- Background processing (safe to re-run on an existing install).
alter table documents add column if not exists status text not null default 'ready';
alter table documents add column if not exists progress text;
alter table documents add column if not exists error text;
