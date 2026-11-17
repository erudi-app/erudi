import { useEffect, useRef } from "react";
import { useLocation } from "react-router-dom";
import { conversationIdFromPath, leaveConversation } from "../utils/ephemeralConversations";

/**
 * Deletes an ephemeral conversation when the user leaves it (#428): when the
 * conversation id in the route changes -- another conversation, the Chat page,
 * any other page. Mounted once inside the router, so leaving is read from the
 * route itself and never from a page unmounting (the conversation page stays
 * mounted from one conversation to the next). A re-render on the same route
 * changes nothing; `leaveConversation` is a no-op for a conversation that is
 * not ephemeral or was already deleted. Renders nothing.
 */
export default function EphemeralConversationGuard() {
  const { pathname } = useLocation();
  const conversationId = conversationIdFromPath(pathname);
  const previousRef = useRef(conversationId);

  useEffect(() => {
    const previous = previousRef.current;
    previousRef.current = conversationId;
    if (previous !== null && previous !== conversationId) {
      leaveConversation(previous);
    }
  }, [conversationId]);

  return null;
}
