"""
Notification utilities for sending SMS updates on batch job status.
"""

import os
import logging
from typing import Optional, Union, Dict, Any

logger = logging.getLogger(__name__)

# Check for Twilio library
try:
    from twilio.rest import Client
    import datetime
    import threading
    import time
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logger.warning("Twilio library not available. SMS notifications will not be sent.")

# Environment variables for Twilio configuration
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER')
TWILIO_TO_PHONE_NUMBER_1 = os.environ.get('TWILIO_TO_PHONE_NUMBER_1')
TWILIO_TO_PHONE_NUMBER_2 = os.environ.get('TWILIO_TO_PHONE_NUMBER_2')
TWILIO_TO_PHONE_NUMBER_3 = os.environ.get('TWILIO_TO_PHONE_NUMBER_2')

def _is_twilio_configured() -> bool:
    """Check if Twilio is properly configured."""
    if not TWILIO_AVAILABLE:
        return False
    
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, TWILIO_TO_PHONE_NUMBER_1]):
        logger.warning("Twilio environment variables not fully configured. SMS notifications will not be sent.")
        return False
    
    return True

def _send_sms(body: str, to_phone: Optional[str] = None) -> bool:
    """
    Send an SMS message using Twilio API.
    
    Args:
        body: The message body text
        to_phone: Target phone number (defaults to TWILIO_TO_PHONE_NUMBER_1)
        
    Returns:
        Boolean indicating success or failure
    """
    if not _is_twilio_configured():
        logger.warning(f"SMS not sent (Twilio not configured): {body}")
        return False
    
    
    # Check if it's quiet hours (between 10 PM and 7 AM)
    now = datetime.datetime.now().time()
    quiet_hours = (now >= datetime.time(22, 0) or now < datetime.time(7, 0))
    
    # Check if message should be scheduled based on content
    is_completion_message = ('status: completed' in body.lower() or 'results saved' in body.lower())
    
    # Send to primary number regardless of time
    success = True
    primary_number = to_phone or TWILIO_TO_PHONE_NUMBER_1
    
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=body,
            from_=TWILIO_PHONE_NUMBER,
            to=primary_number
        )
        logger.info(f"SMS sent successfully to primary number, SID: {message.sid}")
    except Exception as e:
        logger.error(f"Failed to send SMS to primary number: {str(e)}")
        success = False
    
    # # Handle numbers 2 and 3
    # ##if not to_phone:  # Only send to multiple numbers if no specific number was provided
    # for recipient in [TWILIO_TO_PHONE_NUMBER_2, TWILIO_TO_PHONE_NUMBER_3]:
    #     if not recipient:
    #         continue
            
    #     # During quiet hours
    #     if quiet_hours:
    #         # Schedule completion messages for 7:01 AM
    #         if is_completion_message:
    #             def send_scheduled(phone):
    #                 # Calculate time until 7:01 AM
    #                 now = datetime.datetime.now()
    #                 next_morning = now.replace(hour=7, minute=1, second=0, microsecond=0)
    #                 if now.time() >= datetime.time(7, 1):
    #                     next_morning += datetime.timedelta(days=1)
                    
    #                 seconds_to_wait = (next_morning - now).total_seconds()
    #                 time.sleep(seconds_to_wait)
                    
    #                 try:
    #                     client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    #                     message = client.messages.create(
    #                         body=body,
    #                         from_=TWILIO_PHONE_NUMBER,
    #                         to=phone
    #                     )
    #                     logger.info(f"Scheduled SMS sent to {phone}, SID: {message.sid}")
    #                 except Exception as e:
    #                     logger.error(f"Failed to send scheduled SMS to {phone}: {str(e)}")
                
    #             # Create and start thread for scheduled message
    #             thread = threading.Thread(target=send_scheduled, args=(recipient,))
    #             thread.daemon = True
    #             thread.start()
    #             logger.info(f"SMS scheduled for 7:01 AM to {recipient}")
    #     else:
    #         # Send immediately if not quiet hours
    #         try:
    #             message = client.messages.create(
    #                 body=body,
    #                 from_=TWILIO_PHONE_NUMBER,
    #                 to=recipient
    #             )
    #             logger.info(f"SMS sent successfully to {recipient}, SID: {message.sid}")
    #         except Exception as e:
    #             logger.error(f"Failed to send SMS to {recipient}: {str(e)}")
    #             # Don't change overall success status for secondary numbers
    
    return success

def notify_batch_submitted(batch_id: str, task_name: str, num_items: int) -> bool:
    """
    Send notification when a batch job is submitted.
    
    Args:
        batch_id: The ID of the submitted batch
        task_name: Name of the task (e.g., "AAE to SAE translation")
        num_items: Number of items in the batch
        
    Returns:
        Boolean indicating success or failure
    """
    message = f"🚀 {task_name} job submitted\n" \
              f"Items: {num_items}\n" \
              f"Status: Submitted"

    #   f"Batch ID: {batch_id}\n" \
              
    return _send_sms(message)

def notify_batch_status(batch_id: str, task_name: str, status: str, 
                       progress: Union[Dict[str, Any], None] = None) -> bool:
    """
    Send notification about batch job status.
    
    Args:
        batch_id: The ID of the batch
        task_name: Name of the task
        status: Current status of the batch
        progress: Progress information dictionary
        
    Returns:
        Boolean indicating success or failure
    """
    # Always send notifications for completed, failed, cancelled, expired
    if status.lower() in ['completed', 'failed', 'cancelled', 'expired']:
        should_notify = True
    # For in-progress, check progress percentage
    elif status.lower() == 'in_progress' and progress:
        total = progress.get('total', 0)
        completed = progress.get('completed', 0)
        
        # Always notify if there's progress data
        should_notify = True
    else:
        # Other statuses (like submitted)
        should_notify = True
    
    # Skip notification if we decided not to send
    if not should_notify:
        return False
    
    # Format progress info
    progress_text = ""
    if progress:
        total = progress.get('total', 0)
        completed = progress.get('completed', 0) 
        failed = progress.get('failed', 0)
        
        if total > 0:
            percent = (completed / total) * 100
            progress_text = f"\nProgress: {completed}/{total} ({percent:.1f}%)"
            if failed > 0:
                progress_text += f", Failed: {failed}"
    
    # Status emoji
    emoji = "🔄"  # default: in progress
    if status.lower() == 'completed':
        emoji = "✅"
    elif status.lower() in ['failed', 'cancelled', 'expired']:
        emoji = "❌"
    
    message = f"{emoji} {task_name} job update\n" \
              f"Status: {status}{progress_text}"
    
    #   f"Batch ID: {batch_id}\n" \
              
    return _send_sms(message)

def notify_results_saved(task_name: str, num_items: int, output_path: str, dataset_url: str = None) -> bool:
    """
    Send notification when batch results are saved.
    
    Args:
        task_name: Name of the task
        num_items: Number of results saved
        output_path: Where results were saved
        dataset_url: Optional URL to Hugging Face dataset if uploaded
        
    Returns:
        Boolean indicating success or failure
    """
    message = f"💾 {task_name} results saved\n" \
              f"Items: {num_items}\n" \
              f"Output: {os.path.basename(output_path)}"
    
    if dataset_url:
        message += f"\n🤗 View dataset: {dataset_url}"
              
    return _send_sms(message)

def notify_error(task_name: str, error_message: str) -> bool:
    """
    Send notification about an error.
    
    Args:
        task_name: Name of the task
        error_message: Description of the error
        
    Returns:
        Boolean indicating success or failure
    """
    message = f"❌ {task_name} error\n" \
              f"Error: {error_message}"
              
    return _send_sms(message)