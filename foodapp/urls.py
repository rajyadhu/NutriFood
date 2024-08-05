from django.urls import path
import foodapp.views

urlpatterns = [
    path('', foodapp.views.home, name='home'),
    path('home', foodapp.views.home, name='home'),
    path('login',foodapp.views.login, name = 'login'),
    path('user_reg',foodapp.views.userregister, name = 'user_reg'),
    path('up_pro_usr', foodapp.views.up_pro_usr, name='up_pro_usr'),
    path('admin_reg',foodapp.views.adminregister, name = 'admin_reg'),
    path('dietician_reg', foodapp.views.dietician_reg, name='dietician_reg'),
    path('admin_home', foodapp.views.adminHome, name='admin_home'),
    path('user_home', foodapp.views.userHome, name='user_home'),
    path('dietician_home', foodapp.views.dietician_home, name='dietician_home'),
    path('logout', foodapp.views.logout, name='logout'),

    path('fd_img_cla_usr', foodapp.views.fd_img_cla_usr, name='fd_img_cla_usr'),
    path('prdct_user', foodapp.views.prdct_user, name='prdct_user'),
    path('delete_predict_user/<id>', foodapp.views.delete_predict_user, name='delete_predict_user'),

    path('fd_img_cla_adm', foodapp.views.fd_img_cla_adm, name='fd_img_cla_adm'),
    path('cr_new_diet_pln', foodapp.views.cr_new_diet_pln, name='cr_new_diet_pln'),
    path('cons_diet_usr', foodapp.views.cons_diet_usr, name='cons_diet_usr'),

    path('m_m2/<id>', foodapp.views.m_m2, name='m_m2'),
    path('del_msg_student/<id>', foodapp.views.del_msg_student, name='del_msg_student'),
    path('reply_msg_student/<id>', foodapp.views.reply_msg_student, name='reply_msg_student'),
    path('sent_msg_student', foodapp.views.sent_msg_student, name='sent_msg_student'),

    path('m_m', foodapp.views.m_m, name='m_m'),
    path('del_msg_admin/<id>', foodapp.views.del_msg_admin, name='del_msg_admin'),
    path('reply_msg_admin/<id>', foodapp.views.reply_msg_admin, name='reply_msg_admin'),
    path('sent_msg_admin', foodapp.views.sent_msg_admin, name='sent_msg_admin'),

    path('already_sent_usr', foodapp.views.already_sent_usr, name='already_sent_usr'),
    path('already_sent_msg_diet', foodapp.views.already_sent_msg_diet, name='already_sent_msg_diet'),

    path('already_sent_usr', foodapp.views.already_sent_usr, name='already_sent_usr'),
    path('already_sent_msg_diet', foodapp.views.already_sent_msg_diet, name='already_sent_msg_diet'),
    path('bmi/', foodapp.views.bmi, name='bmi'),

    path('liv_chat_usr', foodapp.views.liv_chat_usr, name='liv_chat_usr'),
    path('send', foodapp.views.send, name='send'),
    path('getMessages', foodapp.views.getMessages, name='getMessages'),
    path('clr_cht', foodapp.views.clr_cht, name='clr_cht'),

    path('liv_chat_doct', foodapp.views.liv_chat_doct, name='liv_chat_doct'),
    path('send1', foodapp.views.send1, name='send1'),
    path('getMessages1', foodapp.views.getMessages1, name='getMessages1'),
    path('clr_cht1', foodapp.views.clr_cht1, name='clr_cht1'),

    path('cr_model_chat_bott', foodapp.views.cr_model_chat_bott, name='cr_model_chat_bott'),
    path('cht_chat_bott', foodapp.views.cht_chat_bott, name='cht_chat_bott'),

    path('liv_chat_ai_assi', foodapp.views.liv_chat_ai_assi, name='liv_chat_ai_assi'),
    path('send2', foodapp.views.send2, name='send2'),
]
