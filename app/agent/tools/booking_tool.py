from langchain.tools import tool
from pydantic import BaseModel

_bookings = []

class BookingInput(BaseModel):
    details: str

@tool(args_schema=BookingInput)
def booking_tool(details: str) -> str:
    """
    Use this tool when the user wants to book, reserve, schedule,
    or make an appointment. Input should be a description of what
    the user wants to book, including date and time if mentioned.
    """
    booking_id = len(_bookings) + 1
    booking = {
        "id": booking_id,
        "details": details,
        "status": "confirmed"
    }
    _bookings.append(booking)

    return (
        f"Booking confirmed! ID: #{booking_id}. "
        f"Details: {details}. We will follow up shortly."
    )