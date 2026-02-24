import streamlit as st
import os
from auth_config import get_authenticator # Assuming this is where it comes from


def setup_page():
    """
    Acts as a gatekeeper for all pages.
    Loads CSS, handles authentication, sets the logo, and adds a logout button.
    Returns True if the user is authenticated, False otherwise.
    """
    # 1. Load our custom CSS
    try:
        with open("assets/style.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

    # 2. Initialize the authenticator
    authenticator, config = get_authenticator()

    # 3. Render the login widget
    try:
        authenticator.login()
    except Exception as e:
        st.error(e)

    # 4. Check the authentication status
    auth_status = st.session_state.get("authentication_status")

    if auth_status:
        # --- USER IS LOGGED IN ---
        user_id = st.session_state.get("username", "cbc_admin")
        logo_path = f"user_profile_logos/{user_id}.svg"

        # Safely set the logo (fallback to admin if file is missing)
        if os.path.exists(logo_path):
            st.logo(logo_path, size="large")
        else:
            st.logo("user_profile_logos/cbc_admin.svg", size="large")

        # Put a logout button at the bottom of the sidebar automatically
        with st.sidebar:
            # st.divider()
            authenticator.logout("Uitloggen", "sidebar")

            st.markdown(
                """
                <div class='sidebar-footer'>
                    <a href='https://www.circulairebaggerconsortium.nl/' target='_blank'>
                        Circulair Bagger Consortium
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )

        return True

    elif auth_status is False:
        # --- LOGIN FAILED ---
        st.error('Gebruikersnaam/wachtwoord incorrect')
        return False
"""
    elif auth_status is None:
        # --- NOT LOGGED IN YET ---
        st.warning('Please enter your credentials to access this portal.')
        return False
""";
