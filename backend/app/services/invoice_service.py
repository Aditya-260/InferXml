"""
Invoice PDF Generator — Creates professional invoices for payment records.
"""
import io
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_invoice_pdf(payment, user):
    """
    Generate a professional invoice PDF for a captured payment.

    Args:
        payment: Payment model instance (must have status='captured')
        user: User model instance

    Returns:
        bytes: PDF content as bytes
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Colours ──
    DARK = (30, 30, 36)
    WHITE = (255, 255, 255)
    GREEN = (16, 185, 129)
    GRAY = (160, 160, 170)
    LIGHT_BG = (245, 245, 250)

    company_name = os.getenv('COMPANY_NAME', 'InferX ML')
    company_address = os.getenv('COMPANY_ADDRESS', 'Pune, Maharashtra, India')
    company_email = os.getenv('COMPANY_EMAIL', 'billing@inferx.dev')
    company_gstin = os.getenv('COMPANY_GSTIN', '')

    amount_inr = payment.amount / 100  # paise → INR
    invoice_number = f"INV-{payment.id:06d}"
    invoice_date = payment.updated_at or payment.created_at or datetime.utcnow()

    # ── Header band ──
    pdf.set_fill_color(*DARK)
    pdf.rect(0, 0, 210, 65, 'F')

    # Logo
    logo_path = '/app/app/static/logo.png'
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=15, y=10, w=20)
        text_y = 35
    else:
        text_y = 12

    # Company name
    pdf.set_text_color(*WHITE)
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_xy(15, text_y)
    pdf.cell(100, 10, company_name, new_x="LMARGIN", new_y="NEXT")

    # INVOICE label
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(*GREEN)
    pdf.set_xy(145, 12)
    pdf.cell(50, 10, 'INVOICE', align='R')

    # Invoice number below label
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(145, 22)
    pdf.cell(50, 6, invoice_number, align='R')

    pdf.set_xy(145, 28)
    pdf.cell(50, 6, f"Date: {invoice_date.strftime('%d %b %Y')}", align='R')

    # ── Bill To / From ──
    y = 75

    # From
    pdf.set_text_color(*GRAY)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_xy(15, y)
    pdf.cell(80, 5, 'FROM')
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(*DARK)
    pdf.set_xy(15, y + 6)
    pdf.cell(80, 5, company_name)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(*GRAY)
    pdf.set_xy(15, y + 12)
    pdf.cell(80, 5, company_address)
    pdf.set_xy(15, y + 18)
    pdf.cell(80, 5, company_email)
    if company_gstin:
        pdf.set_xy(15, y + 24)
        pdf.cell(80, 5, f"GSTIN: {company_gstin}")

    # Bill To
    pdf.set_text_color(*GRAY)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_xy(120, y)
    pdf.cell(75, 5, 'BILL TO', align='R')
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(*DARK)
    pdf.set_xy(120, y + 6)
    pdf.cell(75, 5, user.username, align='R')
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(*GRAY)
    pdf.set_xy(120, y + 12)
    pdf.cell(75, 5, user.email, align='R')

    # ── Table ──
    table_y = y + 40

    # Table header
    pdf.set_fill_color(*DARK)
    pdf.set_text_color(*WHITE)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_xy(15, table_y)
    pdf.cell(90, 10, '  Description', fill=True)
    pdf.cell(30, 10, 'Qty', fill=True, align='C')
    pdf.cell(30, 10, 'Rate', fill=True, align='R')
    pdf.cell(35, 10, 'Amount  ', fill=True, align='R')

    # Table row
    row_y = table_y + 10
    pdf.set_fill_color(*LIGHT_BG)
    pdf.set_text_color(*DARK)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_xy(15, row_y)

    plan_name = (payment.plan_granted or 'pro').capitalize()
    description = f"InferX ML {plan_name} Plan - Monthly"

    pdf.cell(90, 10, f"  {description}", fill=True)
    pdf.cell(30, 10, '1', fill=True, align='C')
    pdf.cell(30, 10, f"{payment.currency} {amount_inr:,.2f}", fill=True, align='R')
    pdf.cell(35, 10, f"{payment.currency} {amount_inr:,.2f}  ", fill=True, align='R')

    # ── Totals ──
    totals_y = row_y + 20

    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(*GRAY)
    pdf.set_xy(120, totals_y)
    pdf.cell(40, 7, 'Subtotal:', align='R')
    pdf.set_text_color(*DARK)
    pdf.cell(35, 7, f"{payment.currency} {amount_inr:,.2f}", align='R')

    pdf.set_text_color(*GRAY)
    pdf.set_xy(120, totals_y + 8)
    pdf.cell(40, 7, 'Tax (0%):', align='R')
    pdf.set_text_color(*DARK)
    pdf.cell(35, 7, f"{payment.currency} 0.00", align='R')

    # Grand total
    pdf.set_xy(120, totals_y + 18)
    pdf.set_draw_color(*GREEN)
    pdf.line(120, totals_y + 18, 195, totals_y + 18)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(*GREEN)
    pdf.set_xy(120, totals_y + 20)
    pdf.cell(40, 8, 'Total:', align='R')
    pdf.cell(35, 8, f"{payment.currency} {amount_inr:,.2f}", align='R')

    # ── Payment info ──
    info_y = totals_y + 40

    pdf.set_draw_color(220, 220, 225)
    pdf.line(15, info_y, 195, info_y)

    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(*GRAY)
    pdf.set_xy(15, info_y + 4)
    pdf.cell(40, 5, 'PAYMENT DETAILS')

    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(*DARK)
    pdf.set_xy(15, info_y + 11)
    pdf.cell(40, 5, 'Status:')
    pdf.set_text_color(*GREEN)
    pdf.cell(60, 5, payment.status.upper())

    pdf.set_text_color(*DARK)
    pdf.set_xy(15, info_y + 18)
    pdf.cell(40, 5, 'Payment ID:')
    pdf.set_text_color(*GRAY)
    pdf.cell(80, 5, payment.razorpay_payment_id or 'N/A')

    pdf.set_text_color(*DARK)
    pdf.set_xy(15, info_y + 25)
    pdf.cell(40, 5, 'Order ID:')
    pdf.set_text_color(*GRAY)
    pdf.cell(80, 5, payment.razorpay_order_id or 'N/A')

    pdf.set_text_color(*DARK)
    pdf.set_xy(15, info_y + 32)
    pdf.cell(40, 5, 'Provider:')
    pdf.set_text_color(*GRAY)
    pdf.cell(80, 5, (payment.provider or 'Razorpay').capitalize())

    # ── Footer ──
    pdf.set_y(-25)
    pdf.set_draw_color(220, 220, 225)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(*GRAY)
    pdf.set_y(-20)
    pdf.cell(0, 5, f"This is a computer-generated invoice by {company_name}. No signature required.", align='C')

    return bytes(pdf.output())
