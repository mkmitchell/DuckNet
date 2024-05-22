import os, subprocess
BaseCase = __import__('base_case').BaseCase



class TestDownload(BaseCase):
    @BaseCase.maybe_skip
    def test_download_basic(self):
        if not self.is_chromium() and not self.headed:
            self.skipTest('xdotool does not work with headless firefox for some reason')
        self.open_main(static=False)

        self.send_input_files_from_assets([ "test_image0.jpg", "test_image1.jpg" ])
        self.click('label:contains("test_image1.jpg")')

        root_css = '[filename="test_image1.jpg"]'
        down_css = root_css + ' a.download'

        #download button should be disabled before processing
        assert 'disabled' in self.get_attribute(down_css, 'class')
        #self.click(down_css)  #clicks although disabled, dont use
        from selenium import webdriver
        webdriver.ActionChains(self.driver).click(self.find_element(down_css)).perform()
        self.assert_no_js_errors()
        
        #click on the processing button
        self.click(root_css+' .play.icon')
        #wait until done
        self.wait_for_element_not_visible(root_css+' .dimmer', timeout=6)

        self.click(down_css)
        #send enter key to x11 to confirm the download dialog window
        if not self.is_chromium():  #self.is_firefox()
            self.sleep(1.0)
            subprocess.call('xdotool key Return', shell=True)

        self.assert_downloaded_file('test_image1.jpg.results.zip')

        if self.demo_mode:
            self.sleep(1)

    @BaseCase.maybe_skip
    def test_download_all(self):
        if not self.is_chromium() and not self.headed:
            self.skipTest('xdotool does not work with headless firefox for some reason')
        self.open_main(static=True)

        self.send_input_files_from_assets([
            "test_image0.jpg", "test_image0.jpg.results.zip",
            "test_image1.jpg", "test_image1.jpg.results.zip",
        ])
        
        self.click('[data-tab="detection"] .download-all')
        #send enter key to x11 to confirm the download dialog window
        if not self.is_chromium():  #self.is_firefox()
            self.sleep(1.0)
            subprocess.call('xdotool key Return', shell=True)

        self.assert_downloaded_file('results.zip')
        #TODO: assert donwloaded results are the same as what was uploaded

        if self.demo_mode:
            self.sleep(1)