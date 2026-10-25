
"""Synthetic, offline customer-support tools for the ZentroMart capstone.

The mock order tool gates individual order details behind a simple identity
check. This is an educational simulation, NOT production-ready authentication.
"""

import csv
import os
from datetime import datetime
from uuid import uuid4

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.normpath(os.path.join(_HERE, '..', 'data'))
_TICKETS = os.path.join(_HERE, 'tickets_created.csv')

_TEXT_COLS = {
    'order_id', 'customer_id', 'delivery_pincode', 'return_rrn',
    'cancellation_rrn', 'rto_rrn', 'sku', 'return_sku', 'damage_sku',
    'pincode', 'phone',
}


def _clean(row):
    out = {}

    for key, value in row.items():
        if value == '':
            out[key] = None
        elif value in ('True', 'False'):
            out[key] = value == 'True'
        elif key not in _TEXT_COLS and value.lstrip('-').isdigit():
            out[key] = int(value)
        else:
            out[key] = value

    return out


def _read(name):
    file_path = os.path.join(_DATA, name)

    with open(file_path, encoding='utf-8', newline='') as handle:
        return [_clean(row) for row in csv.DictReader(handle)]


# Load the synthetic dataset.
_ORDERS = {
    order['order_id']: order
    for order in _read('orders.csv')
}

_ITEMS = {}

for item in _read('order_items.csv'):
    _ITEMS.setdefault(item['order_id'], []).append(item)

_CUSTOMERS = {
    customer['customer_id']: customer
    for customer in _read('customers.csv')
}


def lookup_order(order_id: str, verification_value: str = None) -> dict:
    """Return order details only after successful identity verification.

    Accepts either the registered phone's last four digits or the
    delivery pincode. This is an educational simulation, not production
    authentication.
    """

    # Normalize the order ID to digits.
    oid = ''.join(
        ch for ch in str(order_id)
        if '0' <= ch <= '9'
    )

    # Do not reveal order details before verification.
    if (
        verification_value is None
        or str(verification_value).strip() == ''
    ):
        return {
            'verification_required': True,
            'order_id': oid,
            'message': (
                'Ask for the registered phone last 4 digits '
                'OR delivery pincode.'
            )
        }

    # Normalize the supplied verification value.
    proof = ''.join(
        ch for ch in str(verification_value)
        if '0' <= ch <= '9'
    )

    order = _ORDERS.get(oid)

    if order:
        customer = _CUSTOMERS.get(order['customer_id'])

        if customer:
            # Remove spaces and country codes before checking the suffix.
            phone_digits = ''.join(
                ch for ch in str(customer.get('phone') or '')
                if '0' <= ch <= '9'
            )

            pincode_digits = ''.join(
                ch for ch in str(order.get('delivery_pincode') or '')
                if '0' <= ch <= '9'
            )

            # Accept either the phone suffix or the delivery pincode.
            valid = (
                (
                    len(proof) == 4
                    and phone_digits.endswith(proof)
                )
                or (
                    len(proof) == 6
                    and proof == pincode_digits
                )
            )

            if valid:
                return {
                    'found': True,
                    'verified': True,
                    **order,
                    'items': _ITEMS.get(oid, []),
                    'customer_first_name': (
                        customer['name'].split()[0]
                    ),
                    'is_plus': customer['is_plus']
                }

    # Use a generic response for failed verification or unknown orders.
    return {
        'verified': False,
        'verification_required': True,
        'message': 'Verification failed or order not found.'
    }


def create_ticket(
    summary: str,
    order_id: str = None,
    reason: str = None,
    intent: str = None
) -> dict:
    """Create a synthetic human-handover ticket."""

    if not summary or len(summary.strip()) < 20:
        return {
            'created': False,
            'error': (
                'summary too short: include issue, checks made, '
                'and escalation reason'
            )
        }

    ticket_id = 'TKT' + uuid4().hex

    row = {
        'ticket_id': ticket_id,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'order_id': order_id or '',
        'intent': intent or '',
        'reason': reason or '',
        'summary': summary
    }

    fresh = not os.path.exists(_TICKETS)

    with open(
        _TICKETS, 'a', encoding='utf-8', newline=''
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(row.keys())
        )

        if fresh:
            writer.writeheader()

        writer.writerow(row)

    return {
        'created': True,
        'ticket_id': ticket_id
    }
