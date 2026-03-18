import streamlit as st

def acknowledgements_page():
    st.title("🙏 Acknowledagements")
    
    # 创建两列布局
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
                    
        
        At the completion of this project, I would like to express my sincere gratitude to Dr. Mano. Previously, I always disliked and was even afraid of coding, knowing very little about various programs. However, through your course, I have learned a great deal. You not only gave me a satisfied grade but, more importantly, you affirmed my efforts, provided valuable suggestions, and helped me rediscover the joy and confidence in programming, as well as clarify my goals.

        This week's coursework was not easy for me, and I had to spend extra time outside of class to fully comprehend the material. Yet, this process made me feel truly fulfilled. I still encounter many challenges now, but I am determined to overcome them and move steadily toward my goals.

        Once again, thank you very much, Dr. Mano. I would also like to extend my thanks to Teacher Zhang for his guidance. I wish you both every success in your future work and all the best in your endeavors, and Best wishes to you!

        """)
    
