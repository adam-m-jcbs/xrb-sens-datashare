# import gdata.calendar.service

email = 'alex.heger@gmail.com'
password = ''
application_name = 'UMN-StarFit-v1'

# client = gdata.calendar.service.CalendarService()
# client.ClientLogin(email, password, source=application_name)

# client = gdata.calendar.service.CalendarService()
# client.ClientLogin('user@example.com', 'pa$$word', account_type='HOSTED', source='yourCompany-yourAppName-v1')

# Note: By default, the client libraries set the accountType parameter
# is set to HOSTED_OR_GOOGLE. That means ClientLogin will first try to
# authenticate the user's credentials as a Google Apps hosted
# account. If that fails, it will try to authenticate as a Google
# Account. This becomes tricky if user@example.com is both a Google
# Account and a Google Apps account. In that special case, set
# accountType=GOOGLE if the user wishes to use the Google Accounts
# version of user@example.com.

# Later: 
# token = client.GetClientLoginToken() # token is '12345abcde'
# or...
# client.SetClientLoginToken(token)

# Maybe:
# ======
# email = 'user@example.com'
# password = 'pa$$word'
# application_name = 'yourCompany-yourAppName-v1'

# try:
#   client.ClientLogin(email, password, source=application_name)
# except gdata.service.CaptchaRequired:
#   print 'Please visit ' + client.captcha_url
#   answer = raw_input('Answer to the challenge? ')
#   client.ClientLogin(email, password, source=application_name, captcha_token=client.captcha_token, captcha_response=answer)
# except gdata.service.BadAuthentication:
#   exit('Users credentials were unrecognized')
# except gdata.service.Error:
#   exit('Login Error')

# token = 'DQAAMAAAAC8WjGwikVd8i5jNcEKXfP7EKEcoHjq6EcAbPPuY2Vr84YfoaSEOWWQUnsS2mT97iJPxZ3Zu5p9gxA5BXfGwyz31K_dbreQtUdK-F_TdpSb7Nqk1Rj8wgQfB7kPTeN95ya7KDfQ7gEdtFJ2OAxdCrk9PjIsKytNtq4I0jkkOn0hUfxNtgJC41eK4kavkakun5mm_2GQEO5pb2RZ8FnpIMWJK8YTh8av2uX-pFH-25lXar0jheteMrt3_duwuOa-HU'


try: 
    from xml.etree import ElementTree
except ImportError:  
    from elementtree import ElementTree
import gdata.spreadsheet.service
import gdata.service
import atom.service
import gdata.spreadsheet
import atom

import string

def get_client():
    gd_client = gdata.spreadsheet.service.SpreadsheetsService()
    gd_client.email = email
    gd_client.password = password
    gd_client.source = application_name
    gd_client.ProgrammaticLogin()
    return gd_client

def PrintFeed(feed):
    for i, entry in enumerate(feed.entry):
        if isinstance(feed, gdata.spreadsheet.SpreadsheetsCellsFeed):
            print('%s %s\n' % (entry.title.text, entry.content.text))
        elif isinstance(feed, gdata.spreadsheet.SpreadsheetsListFeed):
            print('%s %s %s' % (i, entry.title.text, entry.content.text))
            # Print this row's value for each column (the custom dictionary is
            # built from the gsx: elements in the entry.) See the description of
            # gsx elements in the protocol guide.
            print('Contents:')
            for key in entry.custom:
                print('  %s: %s' % (key, entry.custom[key].text))
            print()
        else:
            print('%s %s\n' % (i, entry.title.text))


def PromptForSpreadsheet(gd_client):
    # Get the list of spreadsheets
    feed = gd_client.GetSpreadsheetsFeed()
    PrintFeed(feed)
    input = input('\nSelection: ')
    return feed.entry[string.atoi(input)].id.text.rsplit('/', 1)[1]

def PromptForWorksheet(gd_client, key):
    # Get the list of worksheets
    feed = gd_client.GetWorksheetsFeed(key)
    PrintFeed(feed)
    input = input('\nSelection: ')
    return feed.entry[string.atoi(input)].id.text.rsplit('/', 1)[1]


# gd_client.UpdateCell(row=2, col=2, inputValue='42', key='0AsWmboMPNT-kdElDbHc0T2ljLVRxZjJ5RlU3Y0dZTnc', wksht_id='od6')

# feed = gd_client.GetCellsFeed('0AsWmboMPNT-kdElDbHc0T2ljLVRxZjJ5RlU3Y0dZTnc', 'od6')

#######################################################################

# CALENDAR

# http://code.google.com/apis/calendar/data/1.0/developers_guide_python.html

import time

def get_calendar_client():
    client = gdata.calendar.service.CalendarService()
    client.ClientLogin(email, password, source=application_name)

    # calendar_service = gdata.calendar.service.CalendarService()
    # calendar_service.email = 'username@domain.com'
    # calendar_service.password = 'mypassword'
    # calendar_service.source = 'Google-Calendar_Python_Sample-1.0'
    # calendar_service.ProgrammaticLogin()


def PrintUserCalendars(calendar_service):
  feed = calendar_service.GetAllCalendarsFeed()
  print(feed.title.text)
  for i, a_calendar in enumerate(feed.entry):
    print('\t%s. %s' % (i, a_calendar.title.text))

def PrintOwnCalendars(calendar_service):
  feed = calendar_service.GetOwnCalendarsFeed()
  print(feed.title.text)
  for i, a_calendar in enumerate(feed.entry):
    print('\t%s. %s' % (i, a_calendar.title.text))

# # Create the calendar
# calendar = gdata.calendar.CalendarListEntry()
# calendar.title = atom.Title(text='Little League Schedule')
# calendar.summary = atom.Summary(text='This calendar contains practice and game times')
# calendar.where = gdata.calendar.Where(value_string='Oakland')
# calendar.color = gdata.calendar.Color(value='#2952A3')
# calendar.timezone = gdata.calendar.Timezone(value='America/Los_Angeles')
# calendar.hidden = gdata.calendar.Hidden(value='false')

# new_calendar = calendar_service.InsertCalendar(new_calendar=calendar)
    
# # Updating existing calendars

# feed = calendar_servicet.GetOwnCalendarsFeed()
# calendar = feed.entry[...]
# # calendar represents a previously retrieved CalendarListEntry
# calendar.title = atom.Title(text='New Title')
# calendar.color = gdata.calendar.Color(value='#A32929')
# updated_calendar = calendar_service.UpdateCalendar(calendar=calendar)

# # Deleting calendars
# To delete a calendar, call the Delete method on the CalendarService
# object, passing the appropriate edit link.

# feed = calendar_servicet.GetOwnCalendarsFeed()
# calendar = feed.entry[...]
# calendar_service.Delete(calendar.GetEditLink().href)

def InsertSingleEvent(calendar_service, title='One-time Tennis with Beth',
                      content='Meet for a quick lesson', where='On the courts',
                      start_time=None, end_time=None):
    event = gdata.calendar.CalendarEventEntry()
    event.title = atom.Title(text=title)
    event.content = atom.Content(text=content)
    event.where.append(gdata.calendar.Where(value_string=where))

    if start_time is None:
      # Use current time for the start_time and have the event last 1 hour
      start_time = time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())
      end_time = time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime(time.time() + 3600))
    event.when.append(gdata.calendar.When(start_time=start_time, end_time=end_time))


#    new_event = calendar_service.InsertEvent(event, '/calendar/feeds/default/private/full')
    new_event = calendar_service.InsertEvent(event, '/calendar/feeds/pnkicahggs7o1524ho62dmorjc%40group.calendar.google.com/private/full')



    print('New single event inserted: %s' % (new_event.id.text))
    print('\tEvent edit URL: %s' % (new_event.GetEditLink().href))
    print('\tEvent HTML URL: %s' % (new_event.GetHtmlLink().href))

    return new_event


# new spreadsheet

import gdata.spreadsheet.text_db

def new_spread_sheet():
    client = gdata.spreadsheet.text_db.DatabaseClient()
    dc = client._GetDocsClient()  
    dc.email = email
    dc.password = password         
    dc.ProgrammaticLogin()  
    db = client.CreateDatabase('google_spreadsheets_db auth sub test')
    return client, db
