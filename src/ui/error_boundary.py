import streamlit as st
import logging

logger = logging.getLogger(__name__)

def with_error_boundary(page_name):
    """
    Decorator to wrap a Streamlit page rendering function with an error boundary.
    Catches exceptions, logs them, and displays a graceful error UI instead of crashing the app.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error rendering {page_name}: {e}", exc_info=True)
                st.error(f"Something went wrong while loading {page_name}.")
                with st.expander("Show Details"):
                    st.exception(e)
        return wrapper
    return decorator
