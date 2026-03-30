-- PostgreSQL Database Initialization Script for Agent RAG
-- Run this script to create the messages table for session management

-- Create messages table for storing conversation history
CREATE TABLE IF NOT EXISTS messages (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'user',
    content TEXT NOT NULL,
    token_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT,
    CONSTRAINT messages_role_check CHECK (role IN ('user', 'assistant', 'system'))
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);

-- Create index for listing distinct sessions
CREATE INDEX IF NOT EXISTS idx_messages_session_distinct ON messages(session_id);

-- Comments for documentation
COMMENT ON TABLE messages IS 'Stores conversation messages for session management. Messages are grouped by session_id.';
COMMENT ON COLUMN messages.id IS 'Unique message identifier (UUID)';
COMMENT ON COLUMN messages.session_id IS 'Session identifier grouping related messages';
COMMENT ON COLUMN messages.role IS 'Message role: user, assistant, or system';
COMMENT ON COLUMN messages.content IS 'Message content/text';
COMMENT ON COLUMN messages.token_count IS 'Optional token count for the message';
COMMENT ON COLUMN messages.created_at IS 'Timestamp when message was created';
COMMENT ON COLUMN messages.metadata_json IS 'Optional metadata in JSON format';

-- Example queries for session management:

-- Get all messages in a session (ordered by time)
-- SELECT * FROM messages WHERE session_id = 'your-session-id' ORDER BY created_at;

-- Get messages formatted for LLM
-- SELECT json_build_object('role', role, 'content', content) FROM messages WHERE session_id = 'your-session-id' ORDER BY created_at;

-- Count messages in a session
-- SELECT COUNT(*) FROM messages WHERE session_id = 'your-session-id';

-- Delete a session (all its messages)
-- DELETE FROM messages WHERE session_id = 'your-session-id';

-- List all session IDs
-- SELECT DISTINCT session_id FROM messages ORDER BY session_id;
