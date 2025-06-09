from django import forms

class OnOffForm(forms.Form):
    on_off_choices = [
        ("좌파", "좌파"),
        ("우파", "우파"),
    ]
    on_off = forms.ChoiceField(
        choices = on_off_choices,
        widget = forms.RadioSelect(attrs = {'class': 'rbtn'}),
        required = False,
        label = "진영",
        initial = "성향",
        )
    