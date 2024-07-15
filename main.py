import streamlit as st
from llm import get_ai_response
from cheeko_order import get_response

st.set_page_config(
    page_title='치코톡',
    page_icon='❤'
)

def check_password():
    if 'order_number' not in st.session_state:
        st.session_state['order_number'] = ''
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        order_number = st.text_input('주문번호를 입력하세요:', type='password')
        if order_number:
            response = get_response(order_number)
            if response.status_code == 200:
                st.session_state['authenticated'] = True
                # Fetch order information from the backend server
                if len(response.json()['data']) != 0:
                    st.session_state['order_info'] = response.json()['data']
                else:
                    st.session_state['order_info'] = {'error': '주문 정보를 가져오는데 실패했습니다.'}
                st.rerun()
            else:
                st.warning('주문번호를 다시 확인해주세요.')
            
check_password()

if st.session_state.get('authenticated', False):
    st.title('치코톡 😎')
    st.caption('배송 관련된 문의를 해보세요!')

    # Display order information
    order_info = st.session_state.get('order_info', {})
    if 'error' in order_info:
        st.error(order_info['error'])
    else:
        st.subheader('주문하신 정보')
        for order in order_info:
            site_name = order.get('site', {}).get('name', 'N/A')
            prod_name = order.get('prod_name', 'N/A')
            order_name = order.get('order_name', 'N/A')
            st.write(f"주문사이트: {site_name}")
            st.write(f"상품명: {prod_name}")
            st.write(f"주문자: {order_name}")
            st.write("---")

    if 'message_list' not in st.session_state:
        st.session_state.message_list = []

    for message in st.session_state.message_list:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_question := st.chat_input(placeholder='무엇이 궁금하신가요?'):
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.message_list.append({
            "role": "user",
            "content": user_question,
        })

        with st.spinner('답변을 생성하는 중입니다'):
            ai_response = get_ai_response(user_question)
            with st.chat_message("ai"):
                ai_message = st.write_stream(ai_response)
                st.session_state.message_list.append({
                    "role": "ai",
                    "content": ai_message,
                })
